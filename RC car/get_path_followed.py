#!/usr/bin/env python
from __future__ import print_function

import roslib
# roslib.load_manifest('racecar')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import tf
import math
import time

center_line = np.loadtxt('rc_car_end/offline-trajectory-tools/center_line_1.csv',delimiter=',')[:,:2]
WAYPOINT_GAP = 0.1
LOOKAHEAD_DIST = 0.6
L = 0.5
max_steering = 0.34
forward_speed = 0.4

def convert_xyzw_to_rpy(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
    
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
    
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
    
        return roll_x, pitch_y, yaw_z # in radians

def pos_callback(data) :
  global slam_pose_x, slam_pose_y, slam_pose_yaw
  slam_pose_x, slam_pose_y = data.pose.position.x, data.pose.position.y
  _,_,slam_pose_yaw = convert_xyzw_to_rpy(data.pose.orientation.x,data.pose.orientation.y,data.pose.orientation.z,data.pose.orientation.w)

def pos_callback_amcl(data) :
  global slam_pose_x, slam_pose_y, slam_pose_yaw
  slam_pose_x, slam_pose_y = data.pose.pose.position.x, data.pose.pose.position.y
  _,_,slam_pose_yaw = convert_xyzw_to_rpy(data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w)

def main(args):
  rospy.init_node('get_path_followed', anonymous=True)
  global tf_listener, pub
  pub = rospy.Publisher('/vesc/joy',Joy)
  global slam_pose_x, slam_pose_y, slam_pose_yaw
  slam_pose_x = 0
  slam_pose_y = 0
  slam_pose_yaw = 0
  amcl_pose = rospy.Subscriber('/amcl_pose',PoseWithCovarianceStamped,pos_callback_amcl)
  
  tf_listener = tf.TransformListener()
  path = []
  i = 0
  prev_pose_x = 0
  prev_pose_y = 0
  start_time = time.time()
  started = False
  while(i<2000) :
    i+=1
    if slam_pose_y>11.40 :
      break
    if (slam_pose_x<0 and not started) or (prev_pose_x==slam_pose_x and prev_pose_y==slam_pose_y) :
      time.sleep(0.02)
      continue
    started = True
    prev_pose_x = slam_pose_x
    prev_pose_y = slam_pose_y
    print(slam_pose_x,slam_pose_y,slam_pose_yaw,i)
    path.append([slam_pose_x,slam_pose_y,slam_pose_yaw,time.time()-start_time])
    np.savetxt('controller_output_with_cbf/iter7_0.csv',np.array(path),delimiter=',')
    # a = input()
    # if a=='q' :
    #     break
  

if __name__ == '__main__':
    main(sys.argv)
