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
import geometry_msgs.msg
import tf
import math
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import torch
import torch.nn as nn
import transforms
from torch.utils.data import DataLoader, Dataset
import resnet as models
import os 
from scipy.stats import norm
import time

############ HEADER #########################
# Change these params
TEST = True
RUN_NO = 6 # Iteration no. For 0th iteration expert controller will be used, else the end-to-end controller will be used
RUN_ID = 0 # If taking multiple runs for the same iteration
LANE_WIDTH = 0.55 # Lane width on one side of centre line. Total lane width = 2*LANE_WIDTH 
stop_dist_thres = 0.8 # Stop if vehicle deviates with more than this much amount from centre line. Set this to little more than LANE_WIDTH 
SAFEGUARD = True # If True, CBF will be used for safeguarding to controller against moving out of lane boundaries
WAYPOINT_GAP = 0.1 # Set equidistant waypoints on the interpolated centre line at this much gap between consecutive points
L = 0.3 # Length of the wheelbase of the vehicle used for kinematic vehicle model
EXPERT_TRACKING = 'pure_pursuit' # Select from 'pure_pursuit', 'mpc'
if EXPERT_TRACKING == 'pure_pursuit' :
  LOOKAHEAD_DIST = 0.8 # Lookahead distance for Pure pursuit
forward_speed = 0.56 # Target speed
lambda_ = 2 # Parameter for CBF
BETA = 0.1 # Parameter for CBF
alpha = 30. # Penalty factor for violation of CBF
WIDTH = 336 # Width of the image from the camera(s)
HEIGHT = 180 # Height of the image from the camera(s)
max_steering = 0.38
max_steering_speed = 1.5
center_line = np.loadtxt('centre_line.csv',delimiter=',')[:,:2] # Center line
race_line = np.loadtxt('raceline3.csv',delimiter=' ')[:,:2] # Race line
USE_GT_STATE = True # If True, GT state obtaimed from localization will be used else the one obtained from DNN will be used

if TEST :
  USE_GT_STATE = False
# IGNORE these for less speeds
DELAY_AWARE = False # For computation time delay compensation
delay = 0. # Shift the initial state by a predicted amount assuming this much computation delay at each step
ACT_DELAY_CONST = 1.0 # For Actuator dynamic delay compensation

########################################

transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406, 0.5), 
                             (0.229, 0.224, 0.225, 0.3))])
curvature_factor = 0.5
x_factor = 1
theta_factor = 0.1
IMAGE_FOLDER = 'run'+str(RUN_NO)+'_images'
model_path = 'saved_models_iter'+str(RUN_NO) 
if SAFEGUARD : 
  model_path_prev = 'saved_models_iter_cbf'+str(RUN_NO-1)
else :
  model_path_prev = 'saved_models_iter_cbf'+str(RUN_NO-1)
start_time =0


import cubic_spline_planner as csp
import mpc
rx,ry,ryaw,rk,rs = csp.calc_spline_course(race_line[:,0],race_line[:,1],ds=WAYPOINT_GAP)
race_line = np.array([rx,ry]).T

if EXPERT_TRACKING == 'mpc' :
  def smooth_yaw(yaw):
      for i in range(len(yaw) - 1):
          dyaw = yaw[i + 1] - yaw[i]

          while dyaw >= math.pi / 2.0:
              yaw[i + 1] -= math.pi * 2.0
              dyaw = yaw[i + 1] - yaw[i]

          while dyaw <= -math.pi / 2.0:
              yaw[i + 1] += math.pi * 2.0
              dyaw = yaw[i + 1] - yaw[i]

      return yaw

  ryaw = smooth_yaw(ryaw)

def has_crossed_end_line(x,y) :
  # print("Received ", x,y)
  if TEST :
    return False
  return (y>10.7)

def get_optimal_control(steer_ref,steer_var,v,theta,theta_var,x,x_var,curvature,curvature_var) :
    # theta = -theta
    # print(steer_ref,v,theta,x,curvature)
    if SAFEGUARD==False :
        return steer_ref
    # return steer_ref
    min_steer = -max_steering
    min_cost = 1000000
    steer_var = 1
    steer = np.arange(-max_steering,max_steering,0.01)
    ax = 0
    h_left = LANE_WIDTH/2. - x
    hd_left = -v*math.sin(theta)
    hdd_left = -ax*math.sin(theta)-v**2*math.cos(theta)*steer/L + v**2*curvature
    h_right = LANE_WIDTH/2. + x
    hd_right = v*math.sin(theta)
    hdd_right = ax*math.sin(theta)+v**2*math.cos(theta)*steer/L - v**2*curvature
    costs = (steer-steer_ref)**2/steer_var**2
    var_left = np.sqrt((abs(lambda_**2*x_var))**2 + (v**2*math.sin(theta)*theta_var/L)**2 + (v**2*curvature_var)**2 + (lambda_*(v*math.cos(theta))*theta_var)**2)
    var_right = np.sqrt((abs(lambda_**2*x_var))**2 + (v**2*math.sin(theta)*theta_var/L)**2 + (v**2*curvature_var)**2 + (lambda_*(v*math.cos(theta))*theta_var)**2)
    # if (hdd_left+lambda_*hd_left+lambda_**2*h_left-var_left*norm.ppf(1-BETA)) < 0 :
        # print("Right violation") 
    costs += ((hdd_left+2*lambda_*hd_left+lambda_**2*h_left-var_left*norm.ppf(1-BETA)) < 0)*alpha*(hdd_left+2*lambda_*hd_left+lambda_**2*h_left-var_left*norm.ppf(1-BETA))**2
    # if  :
        # print("Left violation")
    costs += ((hdd_right+2*lambda_*hd_right+lambda_**2*h_right-var_right*norm.ppf(1-BETA)) < 0)*alpha*(hdd_right+2*lambda_*hd_right+lambda_**2*h_right-var_right*norm.ppf(1-BETA))**2
    min_steer = steer[np.argmin(costs)] 
   
    return min_steer

def pos_callback(data) :
  global slam_pose_x, slam_pose_y, slam_pose_yaw
  slam_pose_x, slam_pose_y = data.pose.position.x, data.pose.position.y
  _,_,slam_pose_yaw = convert_xyzw_to_rpy(data.pose.orientation.x,data.pose.orientation.y,data.pose.orientation.z,data.pose.orientation.w)

def pos_callback_amcl(data) :
  global slam_pose_x, slam_pose_y, slam_pose_yaw
  slam_pose_x, slam_pose_y = 0,0#data.pose.pose.position.x, data.pose.pose.position.y
  _,_,slam_pose_yaw = 0,0,0#convert_xyzw_to_rpy(data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w)

def find_min_dist(p) :
    x,y = p[0], p[1]
    dists = (center_line[:,:2]-np.array([[x,y]]))
    dist = dists[:,0]**2 + dists[:,1]**2
    # print(min(dist))
    mini = np.argmin(dist)
    vals = []
    if mini>0 :
        x1,y1 = center_line[mini-1,0],center_line[mini-1,1]
        x2,y2 = center_line[mini,0],center_line[mini,1]
        a,b,c = -(y2-y1), (x2-x1),y2*x1-y1*x2 
        ym = (a*(a*y-b*x)-b*c)/(a**2+b**2)
        xm = (-b*(a*y-b*x)-a*c)/(a**2+b**2)
        if (xm>min(x1,x2) and xm<max(x1,x2)) or (ym>min(y1,y2) and ym<max(y1,y2)) :
          vals.append(abs((a*x+b*y+c)/(math.sqrt(a**2+b**2))))
        else :
          vals.append(min(math.sqrt((x-x1)**2+(y-y1)**2),math.sqrt((x-x2)**2+(y-y2)**2)))
    if mini < len(center_line)-1 :
        x1,y1 = center_line[mini,0],center_line[mini,1]
        x2,y2 = center_line[mini+1,0],center_line[mini+1,1]
        a,b,c = -(y2-y1), (x2-x1),y2*x1-y1*x2 
        ym = (a*(a*y-b*x)-b*c)/((a**2+b**2))
        xm = (-b*(a*y-b*x)-a*c)/(a**2+b**2)
        if (xm>min(x1,x2) and xm<max(x1,x2)) or (ym>min(y1,y2) and ym<max(y1,y2)) :
          vals.append(abs((a*x+b*y+c)/(math.sqrt(a**2+b**2))))
        else :
          vals.append(min(math.sqrt((x-x1)**2+(y-y1)**2),math.sqrt((x-x2)**2+(y-y2)**2)))
    # print("aa : ", min(vals))
    return min(vals)

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

def computeCurvature(p1,p2,p3) :
  a = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
  b = math.sqrt((p1[0]-p3[0])**2 + (p1[1]-p3[1])**2)
  c = math.sqrt((p3[0]-p2[0])**2 + (p3[1]-p2[1])**2)
  dx1 = p2[0] - p1[0]
  dx2 = p3[0] - p1[0]
  dy1 = p2[1] - p1[1]
  dy2 = p3[1] - p1[1]
  area = 0.5 * (dx1 * dy2 - dy1 * dx2)
  return 4*area/(a*b*c)

class controller:
  def __init__(self):
    # self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/front_camera/zed2/zed_node/stereo/image_rect_color",Image,self.callback,queue_size=1)
    self.prevx = 0
    self.prevy = 0
    self.prev_cmd = 0
    self.prevyaw = 0
    self.prevt = 0
    self.traj = []

  def callback(self,data):
    # Get inputs
    if True :
      try:
        cv_image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)

      except CvBridgeError as e:
        print(e)

      # cv_image = 
      (rows,cols,channels) = cv_image.shape
      # if cols > 60 and rows > 60 :
      #   cv2.circle(cv_image, (50,50), 10, 255)
      now = rospy.Time.now()
      # try:
      #     tf_listener.waitForTransform('/map', '/base_link', now, rospy.Duration(4.0))
      #     (trans,rot) = tf_listener.lookupTransform('/turtle2', '/turtle1', now)
      # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
      #     print("Error reading tf")
      (trans, rot) = tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
      _,_,yaw = convert_xyzw_to_rpy(rot[0],rot[1],rot[2],rot[3])
      print("Current time of transform : ",float(str(now))/1e9)
      # print("Current transform :- x =", slam_pose_x,"y =", slam_pose_x,"yaw =",slam_pose_yaw)
      # print("Time from camera : ",data.header.stamp.secs+data.header.stamp.nsecs/1e9)
      # x,y = trans[0],trans[1]
      t = float(str(now))/1e9
      x,y = slam_pose_x, slam_pose_y
      yaw = slam_pose_yaw
      velx,vely,velyaw = (x-self.prevx)/(t-self.prevt), (y-self.prevy)/(t-self.prevt), (yaw-self.prevyaw)/(t-self.prevt)
      self.prevx = x
      self.prevy = y
      self.prevyaw = yaw
      # self.prev_cmd = 0
      if DELAY_AWARE :
        x += velx*delay
        y += vely*delay
        yaw += velyaw*delay
      
    # Reference expert controller to follow racing line
    if not TEST :
      if EXPERT_TRACKING=='mpc' :
        str_val,a_val = mpc.get_mpc_control(x,y,yaw,math.sqrt(velx**2+vely**2)+0.1,[rx,ry,ryaw,rs])
      else : 
        dists = (race_line - np.array([[x,y]]))
        dists = dists[:,0]**2 + dists[:,1]**2
        mini = np.argmin(dists)
        targeti = min((mini + int(LOOKAHEAD_DIST/WAYPOINT_GAP))%dists.shape[0],len(race_line)-1)
        tarx,tary = race_line[targeti,0],race_line[targeti,1]
        diffx,diffy = tarx-x,tary-y
        dx,dy = diffx*math.cos(yaw) + diffy*math.sin(yaw),diffy*math.cos(yaw)-diffx*math.sin(yaw)
        str_val = 2*L*dy/(dx**2+dy**2)
      
      str_val_updated = (str_val-self.prev_cmd)*ACT_DELAY_CONST+self.prev_cmd
      self.prev_cmd = str_val
      steering = max(min(str_val_updated,max_steering),-max_steering) 
      print("Steering = ", steering/max_steering)
      
    cmd = Joy()
    cmd.header.stamp = now
    cmd.buttons = [0,0,0,0,1,0,0,0,0,0,0]
    
    # Pre-process image :-
    if True :
      if RUN_NO != 0 :
        new_image = np.zeros((360,336,4), np.uint8)
        new_image[:180:2,:,:] = cv_image[97:-1,:336,:]
        new_image[1:180:2,:,:] = cv_image[98:,:336,:]//2 + cv_image[97:-1,:336,:]//2
        new_image[180:360:2,:,:] = cv_image[97:-1,336:,:]
        new_image[181:360:2,:,:] = cv_image[98:,336:,:]//2 + cv_image[97:-1,336:,:]//2
        X = transform(new_image[:,:,[2,1,0,3]])# pub.publish(cmd)
        X1 = X[:3,:HEIGHT,:]
        X2 = X[:3,HEIGHT:,:]
        X = torch.cat((X1,X2),0)
        # print(X.shape)
        inpp = X.unsqueeze(0).float().cuda()
    
    # Get predicted control
    if RUN_NO != 0 :
      controls = max(min(float(model(inpp)),max_steering),-max_steering)
    
    # Get GT state
    if USE_GT_STATE :
      dists = (center_line - np.array([[x,y]]))
      dists = dists[:,0]**2 + dists[:,1]**2
      mini = np.argmin(dists)  
      if mini < len(center_line)-1 : 
        closest_angle_vector = center_line[mini+1,:]-center_line[mini,:]
      else :
        closest_angle_vector = center_line[mini,:]-center_line[mini-1,:]
      vec1 = np.array([x,y])-center_line[mini,:2]
      vec2 = np.array(closest_angle_vector)
      val = vec2[0]*vec1[1] - vec2[1]*vec1[0]
      # x_ = math.sqrt(dists[mini])*val/abs(val)
      x_ = find_min_dist([x,y])*val/abs(val)
      print("Lat err : ",x_)
      if mini < len(center_line)-2 : 
        theta = yaw - math.atan2(center_line[mini+1,1]-center_line[mini,1],center_line[mini+1,0]-center_line[mini,0])
      else :
        theta = yaw - math.atan2(center_line[mini,1]-center_line[mini-1,1],center_line[mini,0]-center_line[mini-1,0])

      while theta > math.pi :
        theta -= 2*math.pi
      while theta < -math.pi :
        theta += 2*math.pi
      curvature = computeCurvature(center_line[mini,:],center_line[mini+2,:],center_line[mini+4,:])
      print("State : (", x_, ',',theta,',',curvature,')')
    
    # Get Predicted state
    if not USE_GT_STATE :
      curvature = float(model_safety_1(inpp))*curvature_factor
      x_ = float(model_safety_2(inpp))*x_factor
      theta = float(model_safety_3(inpp))*theta_factor
      while theta > math.pi :
        theta -= 2*math.pi
      while theta < -math.pi :
        theta += 2*math.pi
      print("State : (", x_, ',',theta,',',curvature,')')
    
    if has_crossed_end_line(x,y) or (USE_GT_STATE and abs(x_) > stop_dist_thres):
        traj = np.array(self.traj)
        if SAFEGUARD :
          cbf_txt = 'with_cbf_'
        else :
          cbf_txt = 'without_cbf_'
        np.savetxt('traj_'+cbf_txt+str(RUN_NO)+'_'+str(RUN_ID)+'.csv',traj)
        print("Finished")
        rospy.signal_shutdown("Finished")
        exit(0)

    # Get optimal control given the state and reference control
    if RUN_NO == 0 :
      cmd.axes = [0.,forward_speed,steering/max_steering,steering/max_steering,0.,0.]
    else :
      pred_control = get_optimal_control(controls,1.,forward_speed*2,theta,0.,x_,0.,curvature,0.)
      cmd.axes = [0.,forward_speed,pred_control/max_steering,steering/max_steering,0.,0.]
      print("Predicted steering : ",pred_control/max_steering)
    
    # cmd.axes = [0.,forward_speed,(steering/max_steering),steering/max_steering,0.,0.]
    pub.publish(cmd)
    

def main(args):
  rospy.init_node('controller', anonymous=True, disable_signals=True)
  global tf_listener, pub
  pub = rospy.Publisher('/vesc/joy',Joy)
  tf_listener = tf.TransformListener()
  global slam_pose_x, slam_pose_y, slam_pose_yaw, model, model_safety_1, model_safety_2, model_safety_3
  slam_pose_x = 0
  slam_pose_y = 0
  slam_pose_yaw = 0

  # Load models
  if RUN_NO != 0 :
    model = models.resnet18()
    in_features = model.fc.in_features
    model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(model.fc.in_features, 1))
    model = model.cuda()
    model.eval()
    
    model_safety_1 = models.resnet18()
    model_safety_1.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model_safety_1.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
    model_safety_1 = model_safety_1.cuda()
    model_safety_1.eval()
    # print(model_safety_1)
    
    model_safety_2 = models.resnet18()
    model_safety_2.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model_safety_2.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
    model_safety_2 = model_safety_2.cuda()
    model_safety_2.eval()
    print(model_safety_2)
    
    model_safety_3 = models.resnet18()
    model_safety_3.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model_safety_3.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
    model_safety_3 = model_safety_3.cuda()
    model_safety_3.eval()

    st_dict = torch.load(os.path.join(model_path_prev, 'model-last.ckpt'))
    model.load_state_dict(st_dict)
    
    st_dict = torch.load(os.path.join(model_path_prev, 'model-last-safety-1.ckpt'))
    model_safety_1.load_state_dict(st_dict)
    
    st_dict = torch.load(os.path.join(model_path_prev, 'model-last-safety-2.ckpt'))
    model_safety_2.load_state_dict(st_dict)
    
    st_dict = torch.load(os.path.join(model_path_prev, 'model-last-safety-3.ckpt'))
    model_safety_3.load_state_dict(st_dict)
  
  # slam_pose = rospy.Subscriber('/slam_out_pose',PoseStamped,pos_callback)
  tf_listener.waitForTransform("/map", "/base_link", rospy.Time(), rospy.Duration(4.0))
  global start_time
  start_time = time.time()
  if not TEST :
    amcl_pose = rospy.Subscriber('/amcl_pose',PoseWithCovarianceStamped,pos_callback_amcl,queue_size=1)
  con = controller()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
