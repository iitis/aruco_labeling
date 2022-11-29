# ArUco Labeling

ROS package to auto labeling objects from images based on ArUco markers.

The software can be used to test deep learning systems in real time directly on a robot. All you have to do is stick the markers on the objects, and the robot will automatically detect the objects and create dataset with images. The collected data can be used to valide performance of the different ML models or even to train the models. 

The results are proceeding in IEEE International Conference on Big Data 2022.

Filus K., Sobczak Ł., Domańska J., Domański A., Cupek R., "Real-time testing of vision-based systems for AGVs with ArUco markers", _IEEE International Conference on Big Data_, 2022

## The repository consists of the following files:

- **src/aruco_labeling.py** - the main ROS node that subscribes to video and creates a dataset with images of detected objects.
- **launch/auto_labeling.launch** - a launch file with parameters to launch aruco_labeling node.
- **results/** - a directory with results obtained from aruco_labeling, it contains 2 created datasets.

## Running the code

To run the aruco_labeling node you can just modify the launch file ``launch/auto_labeling.launch`` and run it using the following command:

```sh
roslaunch aruco_labeling auto_labeling.launch
```

## ROS interface

We provide our code as a ROS node for easy use in robotic applications. The ROS interface of the node consists of the following elements:

### Published Topics

- _~aruco_image_ (sensor_msgs/Image) - output video with detected AruCo markers

### Subscribed Topics

- _~camera_in_ (sensor_msgs/Image) - input RGB video, used to searching for fiducials
- _~depth_in_ (sensor_msgs/Image) - input depth video, used to bluring background on images

### Parameters

- _~detection_period_ (float, default: 5.0) - how often the detection should be performed [s]
- _~distance_threshold_ (float, default: 5.0) - skip detection of objects further than threshold [m]
- _~depth_behind_ (float, default: 0.3) - distance behind the marker to blur [m]
- _~depth_ahead_ (float, default: 0.2) - distance ahead the marker to blur [m]
- _~crop_size_ (int, default: 10) - factor of cropping images for dataset
- _~remove_marker_ (bool, default: False) - remove marker from image (set averaged color)
- _~blur_background_ (bool, default: False) - blur image out of marker
- _~crop_object_ (bool, default: False) - crop image to object
- _~output_dir_ (bool, default: False) - directory for the output dataset, create dir if doesn't exist
