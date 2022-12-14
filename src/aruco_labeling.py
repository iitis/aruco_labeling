#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from cv2 import aruco
from threading import Lock, Thread
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import mobilenet
import numpy as np

class Model(Enum):
    MOBILE_NET_V1 = 1

class ClassificationManager:

    def __init__(self, model_id, verbose=0):
        self.model_id = model_id
        self.verbose = verbose
        self.model = None
        if self.model_id == Model.MOBILE_NET_V1:
            self.model = tf.keras.applications.MobileNet(
                input_shape=None,
                alpha=1.0,
                depth_multiplier=1,
                dropout=0.001,
                include_top=True,
                weights="imagenet",
                input_tensor=None,
                pooling=None,
                classes=1000,
                classifier_activation="softmax"
            )
        else:
            raise Exception("Wrong model id!")

    def predict(self, image):
        dim =  (self.model.layers[0].input_shape[0][1], self.model.layers[0].input_shape[0][2])
        x = tf.keras.preprocessing.image.img_to_array(cv2.resize(image, dim, interpolation = cv2.INTER_AREA))
        x = np.expand_dims(x, axis=0)
        if self.model_id == Model.MOBILE_NET_V1:
            x = mobilenet.preprocess_input(x)
            return self.model.predict(x, verbose=self.verbose)

    def get_printable_predictions(self, preds, top=3):
        if self.model_id == Model.MOBILE_NET_V1:
            return mobilenet.decode_predictions(preds, top=top)[0]

class ArucoManager:
    def __init__(self):
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters_create()

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        return corners, ids

    def draw(self, image, corners, ids):
        return aruco.drawDetectedMarkers(image.copy(), corners, ids)

class LabelingNode():
    def __init__(self):
        self.model_id = Model.MOBILE_NET_V1
        self.aruco = ArucoManager()
        self.ml = ClassificationManager(self.model_id, 1)

        self.message_lock = Lock()
        self.thread_executor = ThreadPoolExecutor(5)
        self.ml_future = self.thread_executor.submit(self.preprocess_and_detect, None, None, None)

        self.image = None
        self.depth = None
        self.br = CvBridge()
        self.sub = rospy.Subscriber("~camera_in", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("~depth_in", Image, self.depth_callback)
        self.pub = rospy.Publisher('~aruco_image', Image, queue_size=10)
        self.detection_period = rospy.Duration(rospy.get_param('~detection_period', 5))
        self.last_detection_time = rospy.Time.from_sec(0)
        self.distance_threshold = rospy.get_param('~distance_threshold', 5)
        self.depth_behind = rospy.get_param('~depth_behind', 0.3)
        self.depth_ahead = rospy.get_param('~depth_ahead', 0.2)
        self.crop_size = rospy.get_param('~crop_size', 10)
        self.remove_marker = rospy.get_param('~remove_marker', False)
        self.blur_background = rospy.get_param('~blur_background', False)
        self.crop_object = rospy.get_param('~crop_object', False)
        self.result_filename = None
        self.result_counter = 0
        self.output_dir = Path(rospy.get_param('~output_dir', ""))
        self.output_dir = self.output_dir / datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def image_callback(self, msg):
        with self.message_lock:
            self.image = self.br.imgmsg_to_cv2(msg, "bgr8")

    def depth_callback(self, msg):
        self.depth = self.br.imgmsg_to_cv2(msg, "32FC1")

    def get_distance_to_pixel(self, point):
        if self.depth is None:
            return 0.0

        rows,cols = self.depth.shape
        if point[1] > rows or point[0] > cols:
            return 0.0

        return self.depth[int(point[1]), int(point[0])]

    def spin(self):
        loop_rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.message_lock:
                if self.image is not None:
                    corners, ids = self.aruco.detect(self.image)
                    self.publish_aruco_frame(self.aruco.draw(self.image, corners, ids))
                    if ids is not None and len(ids) > 0:
                        if self.ml_future.done():
                            self.ml_future.result()
                            self.ml_future = self.thread_executor.submit(self.preprocess_and_detect, self.image.copy(), ids.copy(), corners.copy())
                    self.image = None

            loop_rate.sleep()

    def publish_aruco_frame(self, frame):
        if frame is not None:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            msg = self.br.cv2_to_imgmsg(img_rgb, encoding="rgb8")
            msg.header.frame_id = "camera_link"
            self.pub.publish(msg)

    def preprocess_and_detect(self, image, ids, corners):
        if image is None or ids is None or corners is None:
            return False

        # example ids for 1 detected marker: [[0]]
        # example ids for 2 detected markers: [[1][0]]
        if len(ids) == 1 and len(ids[0]) == 1:
            if rospy.Time.now() - self.last_detection_time < self.detection_period:
                return False

            dist = self.get_distance_to_marker(corners[0][0])
            if dist > self.distance_threshold:
                return False

            corner = np.array(corners[0][0])
            if self.remove_marker:
                image = self.preprocess_remove_marker(image, corner)
            if self.blur_background:
                image = self.preprocess_blur_background(image, dist)
            if self.crop_object:
                image = self.preprocess_crop_object(image, corner)

            preds = self.ml.predict(image)
            print(f"Aruco id: {ids[0][0]} - predicted: {self.ml.get_printable_predictions(preds)}")
            classes_id = np.argsort(-preds, axis=1)[0]
            self.store_result(image, ids[0][0], classes_id[0], classes_id[1], classes_id[2], classes_id[3], classes_id[4])
            self.last_detection_time = rospy.Time.now()

        return True

    def store_result(self, image, aruco_id, class_id1, class_id2, class_id3, class_id4, class_id5):
        if self.result_filename is None:
            self.result_filename = f"{self.model_id.name}_labeling.csv"
            self.result_filename = self.output_dir / self.result_filename
            np.savetxt(self.result_filename, [["id,image_name,aruco_id,class_id1,class_id2,class_id3,class_id4,class_id5"]], delimiter=',', fmt='%s')

        image_name = f"{self.result_counter}.jpg"
        cv2.imwrite(str(self.output_dir / image_name), image)
        with open(self.result_filename,'a') as csvfile:
            np.savetxt(csvfile, [[self.result_counter, image_name, aruco_id, class_id1, class_id2, class_id3, class_id4, class_id5]], delimiter=',', fmt='%s')
        self.result_counter += 1

    def preprocess_remove_marker(self, image, corners):
        x = np.array([corners[0][0], corners[1][0], corners[2][0], corners[3][0]])
        y = np.array([corners[0][1], corners[1][1], corners[2][1], corners[3][1]])
        min_x = int(np.amin(x))
        max_x = int(np.amax(x))
        min_y = int(np.amin(y))
        max_y = int(np.amax(y))
        shift = 2
        mean_color = ((image[max_y+shift, max_x+shift, :].astype(int) + 
                       image[min_y-shift, max_x+shift, :].astype(int) + 
                       image[max_y+shift, min_x-shift, :].astype(int) + 
                       image[min_y-shift, min_x-shift, :].astype(int)) / 4).astype(np.uint8, casting='unsafe')
        new_color = (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))
        cv_corners = [np.array([[px, py] for px, py in corners]).astype(int)]
        cv2.drawContours(image, cv_corners, 0, new_color, -1, cv2.LINE_AA)
        return image

    def get_distance_to_marker(self, corners):
        sum = 0
        for i in range(4):
            sum += self.get_distance_to_pixel(corners[i])
        return sum / 4.0 / 1000.0

    def preprocess_blur_background(self, image, dist):
        if image is None or self.depth is None:
            return image

        depth = self.depth
        depth /= 1000
        depth[depth < dist-self.depth_behind] = 0
        depth[depth > dist+self.depth_ahead] = 0
        depth[depth != 0] = 255
        mask = depth.astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        blur = cv2.blur(image, (100,100), 0)
        image[mask==(0,0,0)] = blur[mask==(0,0,0)]
        return image

    def preprocess_crop_object(self, image, corners):
        x = np.array([corners[0][0], corners[1][0], corners[2][0], corners[3][0]])
        y = np.array([corners[0][1], corners[1][1], corners[2][1], corners[3][1]])
        min_x = int(np.amin(x))
        max_x = int(np.amax(x))
        min_y = int(np.amin(y))
        max_y = int(np.amax(y))
        center_x = int((min_x + max_x) / 2)
        center_y = int((min_y + max_y) / 2)
        d_x = ((max_x - min_x)/2)
        d_y = ((max_y - min_y)/2)
        d = max(d_x, d_y)
        crop_size_x = int(self.crop_size*d)
        crop_size_y = int(self.crop_size*d)
        left_x = max(center_x - crop_size_x, 0)
        right_x = min(center_x + crop_size_x, image.shape[1])
        up_y = max(center_y - crop_size_y, 0)
        down_y = min(center_y + crop_size_y, image.shape[0])
        return image[up_y:down_y, left_x:right_x]

if __name__ == '__main__':
    rospy.init_node("aruco_labeling", anonymous=True)
    my_node = LabelingNode()
    my_node.spin()
