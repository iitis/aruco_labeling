#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from cv2 import aruco
from threading import Lock, Thread
import time
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np

class Node():
    def __init__(self):
        self.image = None
        self.br = CvBridge()
        self.sub = rospy.Subscriber("~camera_in", Image, self.image_callback)
        self.pub = rospy.Publisher('~aruco_image', Image, queue_size=10)
        self.message_lock = Lock()
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters_create()
        self.thread_executor = ThreadPoolExecutor(5)
        self.ml_future = self.thread_executor.submit(self.preprocess_and_detect, (None))
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

    def image_callback(self, msg):
        with self.message_lock:
            self.image = self.br.imgmsg_to_cv2(msg, "bgr8")

    def spin(self):
        loop_rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.message_lock:
                if self.image is not None:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
                    if ids is not None and len(ids) > 0:
                        self.draw_and_publish(corners, ids)
                        if self.ml_future.done():
                            self.ml_future.result()
                            self.ml_future = self.thread_executor.submit(self.preprocess_and_detect, (self.image.copy()))
                    self.image = None

            loop_rate.sleep()

    def draw_and_publish(self, corners, ids):
        frame_markers = aruco.drawDetectedMarkers(self.image.copy(), corners, ids)

        if frame_markers is not None:
            img_rgb = cv2.cvtColor(frame_markers, cv2.COLOR_BGR2RGB)
            msg = self.br.cv2_to_imgmsg(img_rgb, encoding="rgb8")
            msg.header.frame_id = "camera_link"
            self.pub.publish(msg)

    def preprocess_and_detect(self, image):
        if image is None:
            return False

        dim =  (self.model.layers[0].input_shape[0][1], self.model.layers[0].input_shape[0][2])
        x = tf.keras.preprocessing.image.img_to_array(cv2.resize(image, dim, interpolation = cv2.INTER_AREA))
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = self.model.predict(x)
        print('Predicted:', decode_predictions(preds, top=3)[0])

        return True

if __name__ == '__main__':
    rospy.init_node("aruco_labeling", anonymous=True)
    my_node = Node()
    my_node.spin()
