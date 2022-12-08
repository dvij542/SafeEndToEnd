# RC car code 

This folder contains code for RC car. Important notes :-
1. We used simple bicycle model for RC car due to lower speed range and limited computation
2. For higher speeds, running MPC may be heavy for running it high frequency on NVIDIA jetson NX, so pur pursuit can be used instead as an expert controller
3. Set the parameters inside run_iter.py, get_dataset.py header accordingly 

# Table of Contents 
   * [ROS installations](#ros-installations)
   * [Install dependencies](#install-dependencies)
   * [Custom required TorchVision download](#custom-required-torchvision-download)
   * [Running instructions](#running-instructions)
      * [Pre map for localization (Using Hector SLAM)](#pre-map-for-localization)
      * [Record center line data](#record-center-line-data)
      * [Calculate racing line (Optional : Skip if you don't require this)](#calculate-racing-line)
      * [Run 0th iteration](#run-0th-iteration)
      * [Record training data](#record-training-data)
      * [Train models](#train-models)
      * [Run an iteration](#run-an-iteration)
   * [Trained model weights on a track](#trained-model-weights-on-a-track)
   * [Instructions for just testing](#instructions-for-just-testing)
   * [Visualize GRADCAM](#visualize-gradcam)

## ROS installations

Install appropriate ROS distro (kinetic/melodic/noetic for ubuntu 16/18/20, full-desktop version with RviZ, rqt_graph etc) from here : [ROS installation](http://wiki.ros.org/ROS/Installation)

Install the following ROS packages : [hector-slam](http://wiki.ros.org/hector_slam/Tutorials/SettingUpForYourRobot), [amcl](http://wiki.ros.org/amcl)

Install cv_bridge for Python3 ROS : [Link](https://idorobotics.com/2020/08/19/setting-up-ros-with-python-3-and-opencv/#:~:text=Install%20cv_bridge%20from%20source%201%20Create%20the%20ROS,3%20Set%20install%3A%20catkin%20config%20--install%20More%20items)

Setup ROS for running Python3 node : [Link](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674)

Assuming the RC car already has the set up catkin_ws workspace

## Install dependencies

Run :-

```
pip install -r requirements.txt
```

OR 

If using conda :-

## Custom required TorchVision download 

As I was facing problems with directly installing TorchVision for AMD64 architecture using pip or conda, I installed it from source and made some changes to the code, hnce download the modified folder from here as it is referenced by the scripts : [Link](https://drive.google.com/drive/folders/1IRGEQ829USxVrm-ez-c_A3hGJk8qwm6v?usp=sharing)

## Running Instructions

### Pre map for localization (Using Hector SLAM)

Manually mark the track (Using tapes/tubes anything that is consistent and assuming it be similar to the test case). Here, we assume constant lane-width for CBF formulation but can be extended to variable lane width by also predicting lane width as a state from DNN. After setting up the environment

Set the parameter sim-time to False in tutoria.launch :-

```
roscd hector_slam_launch/launch
sudo gedit tutorial.launch
```

Run the following command :-
```
roslaunch hector_slam_launch tutorial.launch
```

Check the map visualization on RviZ which must have been launched. When the map seems satisfactory, save the map using :-

```
rosrun map_server map_server -f saved_map
```

### Record center line data

Open 4 terminals/terminator with 4 windows and with the ros workspace file sources in all terminals :-

```
cd catkin_ws
source devel/setup.bash
```

1. Launch RC car preliminaries in 1st terminal :-

```
roslaunch racecar teleop.launch
```

2. Load the saved map :-

```
rosrun map_server map_server saved_map.yaml
```

3. Run AMCL localization :-

```
roslaunch amcl amcl_omni.launch
```

4. Run record_path.py. Manually drive the RC car across the track. Press enter at each waypoint to record a new waypoint into the file 'center_line_recorded.csv'. 

```
python3 record_path.py
```


After the path is recorded, interpolate a spline onto the waypoints to obtain a path with equidistant waypoints in center_line.csv. Set the inter-waypoint distance, DS in interpolate.py

```
python3 interpolate.py
```


### Calculate racing line (Optional : Skip if you don't require this)

Set vehicle lateral and longitudinal acceleration limits, vehicle parameters like friction parameters in global_racetrajectory_optimization/inputs/veh_dyn_info and global_racetrajectory_optimization/params/racecar.ini

Set LANE_WIDTH in global_racetrajectory_optimization/get_track.py and run :-

```
cd global_racetrajectory_optimization
python3 get_track.py
python3 main_globaltraj.py 
```
Raceline will be generate in global_racetrajectory_optimization/outputs/ . Rename it to raceline.csv and move it to 'RC car' directory

### Run 0th iteration

Set parameters in run_iter.py (Descriptions given in code header). If you don't require to run on racing line, just set the racing_line file path same as the centre_line

Open 5 terminals :-

1. Launch RC car preliminaries in 1st terminal :-

```
roslaunch racecar teleop.launch
```

2. Load the saved map :-

```
rosrun map_server map_server saved_map.yaml
```

3. Run AMCL localization :-

```
roslaunch amcl amcl_omni.launch
```

4. Run run_iter.py :-

```
python3 run_iter.py
```

5. Record rosbag file :-

```
rosbag record /tf /tf_static /amcl_pose /front_camera/zed2/zed_node/stereo/image_rect_color -O run_0.bag
```

### Record training data

Finally we get the training data from 0th iteration. Set RUN_NO=0 in get_dataset.py header. Open 2 terminals with sourced ROS files and run the following commands in both :-

1. Collect training images with labels in the file name :-
```
python3 get_dataset.py
```

2. Run the recorded rosbag file :-
```
rosbag play run_0.bag --clock
```

The newly collected images would appear in run0_images. Eliminate repeated images at the beginning where the RC car starts and at the end

### Train models

To train new state and control prediction models for iter 0, set RUN_NO=0 in train.py header and run :-

```
python3 train.py
```

### Run an iteration

After you have the trained the model for 0th iteration, follow these steps repeatedly for running the RC car for subsequent iterations. Repeat the following steps for n iterations

#### Run RC car for (i)th iteration :-

Set RUN_NO = 0 in run_iter.py (Modify paremeters in run_iter.py like the vehicle model parameters, expert controller algo (pure pursuit/mpc) etc) 
Open 5 terminals :-

1. Launch RC car preliminaries in 1st terminal :-

```
roslaunch racecar teleop.launch
```

2. Load the saved map :-

```
rosrun map_server map_server saved_map.yaml
```

3. Run AMCL localization :-

```
roslaunch amcl amcl_omni.launch
```

4. Run run_iter.py :-

```
python3 run_iter.py
```

5. Record rosbag file :-

```
rosbag record /tf /tf_static /amcl_pose /front_camera/zed2/zed_node/stereo/image_rect_color -O run_{i}.bag
```

#### Get training data for (i)th iteration :-

Set RUN_NO = i in get_dataset.py (Modify paremeters in get_dataset.py like the vehicle model parameters, expert controller algo (pure pursuit/mpc) etc accordingly)


1. Collect training images with labels in the file name :-
```
python3 get_dataset.py
```

2. Run the recorded rosbag file :-
```
rosbag play run_{i}.bag --clock
```

This will make a new folder, run{i}_cbf_images, filter out unwanted images from the folder like the repeated images at the beginning before the vehicle starts moving, the images at the end after the vehicle has crossed the lane boundaries as these would affect the training of the new model. Finally, copy the images from the previous iteration (i-1) i.e. run{i-1}_cbf_images to run{i}_cbf_images with :-

```
cp run{i-1}_cbf_images/* run{i}_cbf_images/* -R
```

#### Train model for (i+1)th iteration

To train new state and control prediction models for iter 0, set RUN_NO=i in train.py header and run :-

```
python3 train.py
```

### Trained model weights on a track

Download the control and state prediction models for all iterations from here :- [Link](https://drive.google.com/drive/folders/1AjvvZxM3FPkj-1y2xrkOPcEPwF2gl-zc?usp=share_link). In case you need to fine tune the weights on a new track, using the last iteration would be most convenient

### Instructions for just testing

For just testing the trained models on a new track (assuming similar environment), just download trained models from [Link](https://drive.google.com/drive/folders/1AjvvZxM3FPkj-1y2xrkOPcEPwF2gl-zc?usp=share_link). Place the downloaded model weight folders for all iterations in RC-car folder. Set RUN_NO = 5 and TEST = True (to not use localization in any way) in run_iter.py header and simply run :-

```
python3 run_iter.py
```

### Visualize GRADCAM

To visualize where the trained CNN is looking at, run vis_gradcam.py. Add test images to 'example_imgs'
