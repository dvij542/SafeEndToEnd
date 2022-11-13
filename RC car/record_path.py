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
  rospy.init_node('record_path', anonymous=True)
  global tf_listener, pub
  pub = rospy.Publisher('/vesc/joy',Joy)
  global slam_pose_x, slam_pose_y, slam_pose_yaw
  slam_pose_x = 0
  slam_pose_y = 0
  slam_pose_yaw = 0
  amcl_pose = rospy.Subscriber('/amcl_pose',PoseWithCovarianceStamped,pos_callback_amcl)
  
  tf_listener = tf.TransformListener()
  # tf_listener.waitForTransform("/map", "/base_link", rospy.Time(), rospy.Duration(4.0))
  path = []
  print("Press enter to record a new point and save it to center_line_recorded.csv, q to quit")
  while(True) :
    print(slam_pose_x,slam_pose_y,slam_pose_yaw)
    path.append([slam_pose_x,slam_pose_y,slam_pose_yaw])
    np.savetxt('center_line_recorded.csv',np.array(path),delimiter=',')
    a = input()
    if a=='q' :
      break
 

if __name__ == '__main__':
    main(sys.argv)
