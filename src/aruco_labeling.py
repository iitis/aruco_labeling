#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from cv2 import aruco
from threading import Lock, Thread
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

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

    def get_printable_predictions(self, preds):
        if self.model_id == Model.MOBILE_NET_V1:
            return mobilenet.decode_predictions(preds, top=3)[0]

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

class Node():
    def __init__(self):
        self.aruco = ArucoManager()
        self.ml = ClassificationManager(Model.MOBILE_NET_V1, 1)

        self.message_lock = Lock()
        self.thread_executor = ThreadPoolExecutor(5)
        self.ml_future = self.thread_executor.submit(self.preprocess_and_detect, None, None)

        self.image = None
        self.br = CvBridge()
        self.sub = rospy.Subscriber("~camera_in", Image, self.image_callback)
        self.pub = rospy.Publisher('~aruco_image', Image, queue_size=10)

    def image_callback(self, msg):
        with self.message_lock:
            self.image = self.br.imgmsg_to_cv2(msg, "bgr8")

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
                            self.ml_future = self.thread_executor.submit(self.preprocess_and_detect, self.image.copy(), ids.copy())
                    self.image = None

            loop_rate.sleep()

    def publish_aruco_frame(self, frame):
        if frame is not None:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            msg = self.br.cv2_to_imgmsg(img_rgb, encoding="rgb8")
            msg.header.frame_id = "camera_link"
            self.pub.publish(msg)

    def preprocess_and_detect(self, image, ids):
        if image is None or ids is None:
            return False

        # example ids for 1 detected marker: [[0]]
        # example ids for 2 detected markers: [[1][0]]
        if len(ids) == 1 and len(ids[0]) == 1:
            preds = self.ml.predict(image)
            print(f"Aruco id: {ids[0][0]} - predicted: {self.ml.get_printable_predictions(preds)}")

        return True

if __name__ == '__main__':
    rospy.init_node("aruco_labeling", anonymous=True)
    my_node = Node()
    my_node.spin()
