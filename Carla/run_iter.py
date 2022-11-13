#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA waypoint follower assessment client script.

./CarlaUE4.sh /Game/Maps/RaceTrack -windowed -carla-server -benchmark -fps=30 -quality-level=Low
python3 module_7.py
A controller assessment to follow a given trajectory, where the trajectory
can be defined using way-points.

STARTING in a moment...
"""
from __future__ import print_function
from __future__ import division

# System level imports
import random
import sys
import os
import argparse
import logging
import time
import math
from turtle import speed
import numpy as np
import csv
import matplotlib.pyplot as plt
import controller2d
import configparser 
from scipy.stats import norm
from carla import image_converter
from PIL import Image
import tqdm
# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
# import live_plotter as lv   # Custom live plotting library
# from carla import Client
# import carla
# from carla            import sensor
from carla.client     import make_carla_client, VehicleControl
from carla.settings   import CarlaSettings
from carla.tcp        import TCPConnectionError
from carla.controller import utils
from carla import sensor
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

WIDTH = 800
HEIGHT = 600
RUN_NO = 1
N_ITERS = 4
BETA = 0.1
SAFEGUARD = True
K_ITERS = 3
VEHICLE_MODEL = 'Dynamic'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
image_size = 256

log_step = 10

transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.CenterCrop(min(HEIGHT,WIDTH)),
        transforms.Resize(image_size),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])


"""
Configurable params
"""
ITER_FOR_SIM_TIMESTEP  = 10     # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 3.00   # game seconds (time before controller start)
TOTAL_RUN_TIME         = 200.00 # game seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER     = 300    # number of frames to buffer after total runtime
NUM_PEDESTRIANS        = 0      # total number of pedestrians to spawn
NUM_VEHICLES           = 0      # total number of vehicles to spawn
SEED_PEDESTRIANS       = 0      # seed for pedestrian spawn randomizer
SEED_VEHICLES          = 0      # seed for vehicle spawn randomizer

WEATHERID = {
    "DEFAULT": 0,
    "CLEARNOON": 1,
    "CLOUDYNOON": 2,
    "WETNOON": 3,
    "WETCLOUDYNOON": 4,
    "MIDRAINYNOON": 5,
    "HARDRAINNOON": 6,
    "SOFTRAINNOON": 7,
    "CLEARSUNSET": 8,
    "CLOUDYSUNSET": 9,
    "WETSUNSET": 10,
    "WETCLOUDYSUNSET": 11,
    "MIDRAINSUNSET": 12,
    "HARDRAINSUNSET": 13,
    "SOFTRAINSUNSET": 14,
}
SIMWEATHER = WEATHERID["CLEARNOON"]     # set simulation weather

PLAYER_START_INDEX = 1      # spawn index for player (keep to 1)
FIGSIZE_X_INCHES   = 8      # x figure size of feedback in inches
FIGSIZE_Y_INCHES   = 8      # y figure size of feedback in inches
PLOT_LEFT          = 0.1    # in fractions of figure width and height
PLOT_BOT           = 0.1    
PLOT_WIDTH         = 0.8
PLOT_HEIGHT        = 0.8
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
# WAYPOINTS_FILENAME = 'racing_line.txt'  # waypoint file to load
WAYPOINTS_FILENAME = 'racetrack_waypoints.txt'  # waypoint file to load
CENTERLINE_FILENAME = 'racetrack_waypoints.txt'
DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0  # some distance from last position before
                                       # simulation ends
                                       
# Path interpolation parameters
INTERP_MAX_POINTS_PLOT    = 10   # number of points used for displaying
                                 # lookahead path
INTERP_LOOKAHEAD_DISTANCE = 20   # lookahead in meters
INTERP_DISTANCE_RES       = 0.01 # distance between interpolated points

# controller output directory
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
                           '/controller_output/'

MAX_STEER = 1
LANE_WIDTH = 11
L = 3.
lambda_ = 5.
alpha = 20.
curvature_factor = 0.005
x_factor = 1
theta_factor = 5
THETA_LIM = 37.*(math.pi/180.)

mass = 800
Cf = 1.2*mass
Cr = 1.2*mass
lf = L/2.
lr = L/2. 
Iz = (mass*lf**2)/2.

class EndtoEnd(models.resnet.ResNet):
    def __init__(self):
        super(EndtoEnd, self).__init__(models.resnet.BasicBlock, [2, 2, 2, 2])
        self.speed_feat_extractor = nn.Linear(2, 64)
        self.final_layer = nn.Linear(64+128, 1)
        
    def forward(self, x, v, vperp):
        # change forward here
        # print(vperp.shape)
        speed_feat = torch.cat([torch.atan2(vperp,v),torch.sqrt(v**2+vperp**2)],dim=1)
        # print(speed_feat.shape)
        x1 = super(EndtoEnd,self).forward(x)
        # print(x1.shape)
        x2 = self.speed_feat_extractor(speed_feat)
        # print(x2.shape)
        x = torch.cat((x1,x2),axis=1)
        # print(x.shape)
        x = self.final_layer(x)
        return x

def computeCurvature(p0, p1, p2) :
    dx1 = p1[0] - p0[0]
    dy1 = p1[1] - p0[1]
    dx2 = p2[0] - p0[0]
    dy2 = p2[1] - p0[1]
    dx3 = p2[0] - p1[0]
    dy3 = p2[1] - p1[1]
    area = 0.5 * (dx1 * dy2 - dy1 * dx2)
    len0 = math.sqrt(dx1**2 + dy1**2)
    len1 = math.sqrt(dx2**2 + dy2**2)
    len2 = math.sqrt(dx3**2 + dy3**2)
    return 4 * area / (len0 * len1 * len2)

def has_crossed_train_line(x,y) :
    return (x>-50)

def get_optimal_control(steer_ref,steer_var,v,theta,theta_var,x,x_var,curvature,curvature_var,v_perp=0.,omega=0.,EPSILON=1e-1) :
    # theta = -theta
    # print(steer_ref,v,theta,x,curvature)
    if SAFEGUARD==False :
        return steer_ref
    min_steer = -MAX_STEER
    min_cost = 1000000
    steer_var = 1
    # VEHICLE_MODEL='Kinematic'
    if VEHICLE_MODEL=='Kinematic' :
        for steer in np.arange(-MAX_STEER,MAX_STEER,0.01) :
            ax = 0
            h_left = LANE_WIDTH/2. - x
            hd_left = -v*math.sin(theta)
            hdd_left = -ax*math.sin(theta)-v**2*math.cos(theta)*steer/L + v**2*curvature
            h_right = LANE_WIDTH/2. + x
            hd_right = v*math.sin(theta)
            hdd_right = ax*math.sin(theta)+v**2*math.cos(theta)*steer/L - v**2*curvature
            cost = (steer-steer_ref)**2/steer_var**2
            var_left = math.sqrt((abs(lambda_**2*x_var))**2 + (v**2*math.sin(theta)*steer/L)**2 + (v**2*curvature_var)**2 + (lambda_*(v*math.cos(theta))*theta_var)**2)
            var_right = math.sqrt((abs(lambda_**2*x_var))**2 + (v**2*math.sin(theta)*steer/L)**2 + (v**2*curvature_var)**2 + (lambda_*(v*math.cos(theta))*theta_var)**2)
            if (hdd_left+lambda_*hd_left+lambda_**2*h_left-var_left*norm.ppf(1-BETA)) < 0 :
                print("Right violation") 
                cost += alpha*(hdd_left+lambda_*hd_left+lambda_**2*h_left-var_left*norm.ppf(1-BETA))**2
            if (hdd_right+lambda_*hd_right+lambda_**2*h_right-var_right*norm.ppf(1-BETA)) < 0 :
                print("Left violation")
                cost += alpha*(hdd_right+lambda_*hd_right+lambda_**2*h_right-var_right*norm.ppf(1-BETA))**2
            if cost < min_cost :
                min_cost = cost
                min_steer = steer
    else :
        v += EPSILON
        for steer in np.arange(-MAX_STEER,MAX_STEER,0.01) :
            ax = 0
            x_dot = v_perp*math.cos(theta)+v*math.sin(theta)
            v_perp_dot = (-2*(Cf+Cr)/(mass*v))*v_perp
            v_perp_dot_dot = (-2*(Cf+Cr)/(mass*v))*v_perp_dot + (2*(Cf+Cr)/(mass*v**2))*ax
            x_dot_dot = v_perp_dot*math.cos(theta)+ax*math.sin(theta)+omega*(-v_perp*math.sin(theta)+v*math.cos(theta))
            omega_dot = (-2*(lf**2*Cf + lr**2*Cr)/(Iz*v))*omega+(2*(lf*Cf/Iz))*steer
            x_dot_dot_dot = v_perp_dot_dot*math.cos(theta) + \
                omega_dot*(-v_perp*math.sin(theta)+v*math.cos(theta)) - \
                omega**2*(x_dot)
            h_theta_left = THETA_LIM - theta
            hd_theta_left = - (omega-v*curvature)
            hdd_theta_left = - (omega_dot-ax*curvature)
            h_theta_right = THETA_LIM + theta
            hd_theta_right = (omega-v*curvature)
            hdd_theta_right = (omega_dot-ax*curvature)
                
            h_left = LANE_WIDTH/2. - x
            hd_left = -x_dot
            hdd_left = -x_dot_dot + v**2*curvature
            hddd_left = -x_dot_dot_dot + 2*v*curvature*ax
            h_right = LANE_WIDTH/2. + x
            hd_right = x_dot
            hdd_right = x_dot_dot - v**2*curvature
            hddd_right = x_dot_dot_dot - 2*v*curvature*ax
            cost = (steer-steer_ref)**2/steer_var**2
            
            if (hdd_theta_left+3*lambda_*hd_theta_left+3*lambda_**2*h_theta_left<0) :
                cost += alpha*(hdd_theta_left+3*lambda_*hd_theta_left+3*lambda_**2*h_theta_left)**2
            if (hdd_theta_right+3*lambda_*hd_theta_right+3*lambda_**2*h_theta_right<0) :
                cost += alpha*(hdd_theta_right+3*lambda_*hd_theta_right+3*lambda_**2*h_theta_right)**2
            var_left = math.sqrt((abs(lambda_**2*x_var))**2 + (v**2*math.sin(theta)*steer/L)**2 + (v**2*curvature_var)**2 + (lambda_*(v*math.cos(theta))*theta_var)**2)
            var_right = math.sqrt((abs(lambda_**2*x_var))**2 + (v**2*math.sin(theta)*steer/L)**2 + (v**2*curvature_var)**2 + (lambda_*(v*math.cos(theta))*theta_var)**2)
            if (hddd_left + 3*lambda_*hdd_left + 3*lambda_**2*hd_left+3*lambda_**3*h_left-var_left*norm.ppf(1-BETA)) < 0 :
                print("Right violation")
                cost += alpha*(hddd_left + 3*lambda_*hdd_left + 3*lambda_**2*hd_left+lambda_**3*h_left-var_left*norm.ppf(1-BETA))**2
            if (hddd_right+3*lambda_*hdd_right+3*lambda_**2*hd_right+lambda_**3*h_right-var_right*norm.ppf(1-BETA)) < 0 :
                print("Left violation")
                cost += alpha*(hddd_right+3*lambda_*hdd_right+3*lambda_**2*hd_right+lambda_**3*h_right-var_right*norm.ppf(1-BETA))**2
            if cost < min_cost :
                min_cost = cost
                min_steer = steer
    return min_steer

def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need.
    """
    settings = CarlaSettings()
    
    # There is no need for non-agent info requests if there are no pedestrians
    # or vehicles.
    get_non_player_agents_info = False
    if (NUM_PEDESTRIANS > 0 or NUM_VEHICLES > 0):
        get_non_player_agents_info = True

    # Base level settings
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=get_non_player_agents_info, 
        NumberOfVehicles=NUM_VEHICLES,
        NumberOfPedestrians=NUM_PEDESTRIANS,
        SeedVehicles=SEED_VEHICLES,
        SeedPedestrians=SEED_PEDESTRIANS,
        WeatherId=SIMWEATHER,
        QualityLevel=args.quality_level)
    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)
    camera1 = sensor.Camera('CameraRGB_video')
    camera1.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera1.set_position(-10.0, 0.0, 6.0)
    camera1.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera1)
    
    return settings

class Timer(object):
    """ Timer Class
    
    The steps are used to calculate FPS, while the lap or seconds since lap is
    used to compute elapsed time.
    """
    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        if self.elapsed_seconds_since_lap() >= self._period_for_lap:
            return True
        else:
            return False

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) /\
                     self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

def get_current_pose(measurement):
    """Obtains current x,y,yaw pose from the client measurements
    
    Obtains the current x,y, and yaw pose from the client measurements.

    Args:
        measurement: The CARLA client measurements (from read_data())

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x   = measurement.player_measurements.transform.location.x
    y   = measurement.player_measurements.transform.location.y
    yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)

    return (x, y, yaw)

def get_start_pos(scene):
    """Obtains player start x,y, yaw pose from the scene
    
    Obtains the player x,y, and yaw pose from the scene.

    Args:
        scene: The CARLA scene object

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x = scene.player_start_spots[0].location.x
    y = scene.player_start_spots[0].location.y
    yaw = math.radians(scene.player_start_spots[0].rotation.yaw)

    return (x, y, yaw)

def send_control_command(client, throttle, steer, brake, 
                         hand_brake=False, reverse=False):
    """Send control command to CARLA client.
    
    Send control command to CARLA client.

    Args:
        client: The CARLA client object
        throttle: Throttle command for the sim car [0, 1]
        steer: Steer command for the sim car [-1, 1]
        brake: Brake command for the sim car [0, 1]
        hand_brake: Whether the hand brake is engaged
        reverse: Whether the sim car is in the reverse gear
    """
    control = VehicleControl()
    # Clamp all values within their limits
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)

    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = hand_brake
    control.reverse = reverse
    client.send_control(control)

def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def store_trajectory_plot(graph, fname):
    """ Store the resulting plot.
    """
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)

    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)

def write_trajectory_file(x_list, y_list, v_list, t_list, steerings_list, throttle_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'trajectory_run'+str(RUN_NO)+'.txt')

    with open(file_name, 'w') as trajectory_file: 
        for i in range(len(x_list)):
            trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f, %6.3f, %6.3f\n' %\
                                  (x_list[i], y_list[i], v_list[i], t_list[i], steerings_list[i], throttle_list[i]))

def exec_waypoint_nav_demo(args):
    """ Executes waypoint navigation demo.
    """
    # client = carla.Client('localhost', 2000)
    # return
    with make_carla_client(args.host, args.port) as client:
        print('Carla client connected.')

        settings = make_carla_settings(args)
        
        # Now we load these settings into the server. The server replies
        # with a scene description containing the available start spots for
        # the player. Here we can provide a CarlaSettings object or a
        # CarlaSettings.ini file as string.
        scene = client.load_settings(settings)
        # world = client.get_world()s
        # Refer to the player start folder in the WorldOutliner to see the 
        # player start information
        player_start = PLAYER_START_INDEX

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print('Starting new episode at %r...' % scene.map_name)
        client.start_episode(player_start)
        # blueprint_library = world.get_blueprint_library()
        # vehicle_bp = random.choice(blueprint_library.filter('vehicle.Mustang.*'))
        # for attr in vehicle_bp :
        #     print(attr.id)
        # exit(0)
        #############################################
        # Load Configurations
        #############################################

        # Load configuration file (options.cfg) and then parses for the various
        # options. Here we have two main options:
        # live_plotting and live_plotting_period, which controls whether
        # live plotting is enabled or how often the live plotter updates
        # during the simulation run.
        config = configparser.ConfigParser()
        config.read(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'options.cfg'))         
        demo_opt = config['Demo Parameters']

        # Get options
        enable_live_plot = demo_opt.get('live_plotting', 'true').capitalize()
        enable_live_plot = enable_live_plot == 'True'
        live_plot_period = float(demo_opt.get('live_plotting_period', 0))

        # Set options
        live_plot_timer = Timer(live_plot_period)

        #############################################
        # Load Waypoints
        #############################################
        # Opens the waypoint file and stores it to "waypoints"
        waypoints_file = WAYPOINTS_FILENAME
        waypoints_np   = None
        with open(waypoints_file) as waypoints_file_handle:
            waypoints = list(csv.reader(waypoints_file_handle, 
                                        delimiter=',',
                                        quoting=csv.QUOTE_NONNUMERIC))
            waypoints_np = np.array(waypoints)
        
        centerline_file = CENTERLINE_FILENAME
        centerline_np   = None
        with open(centerline_file) as waypoints_file_handle:
            centerline = list(csv.reader(waypoints_file_handle, 
                                        delimiter=',',
                                        quoting=csv.QUOTE_NONNUMERIC))
            centerline_np = np.array(centerline)

        # Because the waypoints are discrete and our controller performs better
        # with a continuous path, here we will send a subset of the waypoints
        # within some lookahead distance from the closest point to the vehicle.
        # Interpolating between each waypoint will provide a finer resolution
        # path and make it more "continuous". A simple linear interpolation
        # is used as a preliminary method to address this issue, though it is
        # better addressed with better interpolation methods (spline 
        # interpolation, for example). 
        # More appropriate interpolation methods will not be used here for the
        # sake of demonstration on what effects discrete paths can have on
        # the controller. It is made much more obvious with linear
        # interpolation, because in a way part of the path will be continuous
        # while the discontinuous parts (which happens at the waypoints) will 
        # show just what sort of effects these points have on the controller.
        # Can you spot these during the simulation? If so, how can you further
        # reduce these effects?
        
        # Linear interpolation computations
        # Compute a list of distances between waypoints
        wp_distance = []   # distance array
        for i in range(1, waypoints_np.shape[0]):
            wp_distance.append(
                    np.sqrt((waypoints_np[i, 0] - waypoints_np[i-1, 0])**2 +
                            (waypoints_np[i, 1] - waypoints_np[i-1, 1])**2))
        wp_distance.append(0)  # last distance is 0 because it is the distance
                               # from the last waypoint to the last waypoint

        # Linearly interpolate between waypoints and store in a list
        wp_interp      = []    # interpolated values 
                               # (rows = waypoints, columns = [x, y, v])
        wp_interp_hash = []    # hash table which indexes waypoints_np
                               # to the index of the waypoint in wp_interp
        interp_counter = 0     # counter for current interpolated point index
        for i in range(waypoints_np.shape[0] - 1):
            # Add original waypoint to interpolated waypoints list (and append
            # it to the hash table)
            wp_interp.append(list(waypoints_np[i]))
            wp_interp_hash.append(interp_counter)   
            interp_counter+=1
            
            # Interpolate to the next waypoint. First compute the number of
            # points to interpolate based on the desired resolution and
            # incrementally add interpolated points until the next waypoint
            # is about to be reached.
            num_pts_to_interp = int(np.floor(wp_distance[i] /\
                                         float(INTERP_DISTANCE_RES)) - 1)
            wp_vector = waypoints_np[i+1] - waypoints_np[i]
            wp_uvector = wp_vector / np.linalg.norm(wp_vector)
            for j in range(num_pts_to_interp):
                next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                wp_interp.append(list(waypoints_np[i] + next_wp_vector))
                interp_counter+=1
        # add last waypoint at the end
        wp_interp.append(list(waypoints_np[-1]))
        wp_interp_hash.append(interp_counter)   
        interp_counter+=1

        #############################################
        # Controller 2D Class Declaration
        #############################################
        # This is where we take the controller2d.py class
        # and apply it to the simulator
        controller = controller2d.Controller2D(waypoints)

        #############################################
        # Determine simulation average timestep (and total frames)
        #############################################
        # Ensure at least one frame is used to compute average timestep
        num_iterations = ITER_FOR_SIM_TIMESTEP
        if (ITER_FOR_SIM_TIMESTEP < 1):
            num_iterations = 1

        # Gather current data from the CARLA server. This is used to get the
        # simulator starting game time. Note that we also need to
        # send a command back to the CARLA server because synchronous mode
        # is enabled.
        measurement_data, sensor_data = client.read_data()
        sim_start_stamp = measurement_data.game_timestamp / 1000.0
        # Send a control command to proceed to next iteration.
        # This mainly applies for simulations that are in synchronous mode.
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        # Computes the average timestep based on several initial iterations
        sim_duration = 0
        for i in range(num_iterations):
            # Gather current data
            measurement_data, sensor_data = client.read_data()
            # Send a control command to proceed to next iteration
            send_control_command(client, throttle=0.0, steer=0, brake=1.0)
            # Last stamp
            if i == num_iterations - 1:
                sim_duration = measurement_data.game_timestamp / 1000.0 -\
                               sim_start_stamp  
        
        # Outputs average simulation timestep and computes how many frames
        # will elapse before the simulation should end based on various
        # parameters that we set in the beginning.
        SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
        print("SERVER SIMULATION STEP APPROXIMATION: " + \
              str(SIMULATION_TIME_STEP))
        TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) /\
                               SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER

        #############################################
        # Frame-by-Frame Iteration and Initialization
        #############################################
        # Store pose history starting from the start position
        measurement_data, sensor_data = client.read_data()
        start_x, start_y, start_yaw = get_current_pose(measurement_data)
        print("Player data : ",measurement_data.player_measurements)

        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        x_history     = [start_x]
        y_history     = [start_y]
        yaw_history   = [start_yaw]
        time_history  = [0]
        speed_history = [0]
        steerings_list = [0]
        throttles_list = [0]
        
        if True :
            #############################################
            # Vehicle Trajectory Live Plotting Setup
            #############################################
            # Uses the live plotter to generate live feedback during the simulation
            # The two feedback includes the trajectory feedback and
            # the controller feedback (which includes the speed tracking).
            # lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
            # lp_1d = lv.LivePlotter(tk_title="Controls Feedback")
            
            ###
            # Add 2D position / trajectory plot
            ###
            # trajectory_fig = lp_traj.plot_new_dynamic_2d_figure(
                    # title='Vehicle Trajectory',
                    # figsize=(FIGSIZE_X_INCHES, FIGSIZE_Y_INCHES),
                    # edgecolor="black",
                    # rect=[PLOT_LEFT, PLOT_BOT, PLOT_WIDTH, PLOT_HEIGHT])

            # trajectory_fig.set_invert_x_axis() # Because UE4 uses left-handed 
            #                                    # coordinate system the X
            #                                    # axis in the graph is flipped
            # trajectory_fig.set_axis_equal()    # X-Y spacing should be equal in size

            # # Add waypoint markers
            # trajectory_fig.add_graph("waypoints", window_size=waypoints_np.shape[0],
            #                          x0=waypoints_np[:,0], y0=waypoints_np[:,1],
            #                          linestyle="-", marker="", color='g')
            # # Add trajectory markers
            # trajectory_fig.add_graph("trajectory", window_size=TOTAL_EPISODE_FRAMES,
            #                          x0=[start_x]*TOTAL_EPISODE_FRAMES, 
            #                          y0=[start_y]*TOTAL_EPISODE_FRAMES,
            #                          color=[1, 0.5, 0])
            # # Add lookahead path
            # trajectory_fig.add_graph("lookahead_path", 
            #                          window_size=INTERP_MAX_POINTS_PLOT,
            #                          x0=[start_x]*INTERP_MAX_POINTS_PLOT, 
            #                          y0=[start_y]*INTERP_MAX_POINTS_PLOT,
            #                          color=[0, 0.7, 0.7],
            #                          linewidth=4)
            # # Add starting position marker
            # trajectory_fig.add_graph("start_pos", window_size=1, 
            #                          x0=[start_x], y0=[start_y],
            #                          marker=11, color=[1, 0.5, 0], 
            #                          markertext="Start", marker_text_offset=1)
            # # Add end position marker
            # trajectory_fig.add_graph("end_pos", window_size=1, 
            #                          x0=[waypoints_np[-1, 0]], 
            #                          y0=[waypoints_np[-1, 1]],
            #                          marker="D", color='r', 
            #                          markertext="End", marker_text_offset=1)
            # # Add car marker
            # trajectory_fig.add_graph("car", window_size=1, 
            #                          marker="s", color='b', markertext="Car",
            #                          marker_text_offset=1)

            ###
            # Add 1D speed profile updater
            ###
            # forward_speed_fig =\
            #         lp_1d.plot_new_dynamic_figure(title="Forward Speed (m/s)")
            # forward_speed_fig.add_graph("forward_speed", 
            #                             label="forward_speed", 
            #                             window_size=TOTAL_EPISODE_FRAMES)
            # forward_speed_fig.add_graph("reference_signal", 
            #                             label="reference_Signal", 
            #                             window_size=TOTAL_EPISODE_FRAMES)

            # # Add throttle signals graph
            # throttle_fig = lp_1d.plot_new_dynamic_figure(title="Throttle")
            # throttle_fig.add_graph("throttle", 
            #                       label="throttle", 
            #                       window_size=TOTAL_EPISODE_FRAMES)
            # # Add brake signals graph
            # brake_fig = lp_1d.plot_new_dynamic_figure(title="Brake")
            # brake_fig.add_graph("brake", 
            #                       label="brake", 
            #                       window_size=TOTAL_EPISODE_FRAMES)
            # # Add steering signals graph
            # steer_fig = lp_1d.plot_new_dynamic_figure(title="Steer")
            # steer_fig.add_graph("steer", 
            #                       label="steer", 
            #                       window_size=TOTAL_EPISODE_FRAMES)

            # live plotter is disabled, hide windows
            # if not enable_live_plot:
            #     lp_traj._root.withdraw()
            #     lp_1d._root.withdraw()        

            # Iterate the frames until the end of the waypoints is reached or
            # the TOTAL_EPISODE_FRAMES is reached. The controller simulation then
            # ouptuts the results to the controller output directory.
            haha = True
        reached_the_end = False
        skip_first_frame = True
        closest_index    = 0  # Index of waypoint that is currently closest to
                              # the car (assumed to be the first index)
        closest_index_centre    = 0  # Index of waypoint that is currently closest to
                              # the car (assumed to be the first index)
        closest_distance = 0  # Closest distance of closest waypoint to car
        cmd_steer = 0
        cmd_steer1 = 0
        cmd_throttle = 0
        if VEHICLE_MODEL=='Kinematic' :
            model = models.resnet18()
            in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
            model = model.cuda()
            model.eval()
            model.fc.train()
        else :
            model = EndtoEnd()
            in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 128))
            model = model.cuda()
            model.eval()
            model.fc.train()

        model_safety_1 = models.resnet18()
        model_safety_1.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
        model_safety_1 = model_safety_1.cuda()
        model_safety_1.eval()
        model_safety_1.fc.train()
        # print(model_safety_1)
        
        model_safety_2 = models.resnet18()
        model_safety_2.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
        model_safety_2 = model_safety_2.cuda()
        model_safety_2.eval()
        model_safety_2.fc.train()
        # print(model_safety_2)
        
        model_safety_3 = models.resnet18()
        model_safety_3.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
        model_safety_3 = model_safety_3.cuda()
        model_safety_3.fc.train()
        # print(model_safety_3)
        
        if RUN_NO!=0 :
            model_safety_1.load_state_dict(torch.load(os.path.join(model_path, 'model-last-safety-1.ckpt')))
            st_dict = torch.load(os.path.join(model_path, 'model-last-safety-2.ckpt'))
            # st_dict['fc.1.weight'] = st_dict['fc.weight']
            # st_dict['fc.1.bias'] = st_dict['fc.bias']
            # del st_dict["fc.weight"]
            # del st_dict["fc.bias"]
            model_safety_2.load_state_dict(st_dict)
            
            model_safety_3.load_state_dict(torch.load(os.path.join(model_path, 'model-last-safety-3.ckpt')))
            model_safety_3.eval()
        
            print(model)
            
        
            model.load_state_dict(torch.load(os.path.join(model_path, 'model-last.ckpt')))
        
        if not os.path.exists('run'+str(RUN_NO)+'_images'):
            os.makedirs('run'+str(RUN_NO)+'_images')
        if not os.path.exists('run'+str(RUN_NO)+'_video'):
            os.makedirs('run'+str(RUN_NO)+'_video')

        print("Running for", TOTAL_EPISODE_FRAMES)
        curvature_save = 0
        x_save = 0
        theta_save = 0
        theta = 0
        theta_var = 0
        x = 0
        x_var = 0
        curvature = 0
        curvature_var = 0
        theta_comps = []
        steer_var = 1
        x_comps = []
        curvature_comps = []
        current_x, current_y, current_yaw = 0.,0.,0.
        current_timestamp = 0.
        for frame in tqdm.tqdm(range(TOTAL_EPISODE_FRAMES)):
            # Gather current data from the CARLA server
            measurement_data, sensor_data = client.read_data()
            # print(sensor_data)
            if abs(x_save) > 9.5 or theta_save < -math.pi/2. or theta_save > math.pi/2.:
                break
            # print(img_array.shape)
            # Update pose, timestamp
            prev_x, prev_y, prev_yaw = current_x, current_y, current_yaw
            current_x, current_y, current_yaw = \
                get_current_pose(measurement_data)
            vel_x_num = (current_x-prev_x)/(float(measurement_data.game_timestamp) / 1000.0 - (current_timestamp+WAIT_TIME_BEFORE_START) )
            vel_y_num = (current_y-prev_y)/(float(measurement_data.game_timestamp) / 1000.0 - (current_timestamp+WAIT_TIME_BEFORE_START))
            current_speed = measurement_data.player_measurements.forward_speed
            current_speed_perp = vel_y_num*math.cos(current_yaw) - vel_x_num*math.sin(current_yaw)
            current_speed = vel_x_num*math.cos(current_yaw) + vel_y_num*math.sin(current_yaw)
            current_omega = (current_yaw-prev_yaw)/(float(measurement_data.game_timestamp) / 1000.0 - (current_timestamp+WAIT_TIME_BEFORE_START))
            # print("x,y : ", current_x, current_y, current_yaw, vel_x_num, vel_y_num)
            # print("prev_x,prev_y : ", prev_x, prev_y, (current_x-prev_x), (current_y-prev_y), float(measurement_data.game_timestamp) / 1000.0 - current_timestamp)
            # print("Time : ",float(measurement_data.game_timestamp) / 1000.0)
            print("Numericals : ", current_speed, current_speed_perp,current_omega)
            current_timestamp = float(measurement_data.game_timestamp) / 1000.0

            # Wait for some initial time before starting the demo
            if current_timestamp <= WAIT_TIME_BEFORE_START:
                send_control_command(client, throttle=0.0, steer=0, brake=1.0)
                continue
            else:
                current_timestamp = current_timestamp - WAIT_TIME_BEFORE_START
            
            
            
            # Store history
            if frame%5 == 0 :
                x_history.append(current_x)
                y_history.append(current_y)
                yaw_history.append(current_yaw)
                speed_history.append(current_speed)
                time_history.append(current_timestamp) 
                steerings_list.append(cmd_steer)
                throttles_list.append(cmd_throttle)
                main_image = sensor_data.get('CameraRGB', None)
                video_image = sensor_data.get('CameraRGB_video', None)
                img_array_video = np.array(image_converter.to_rgb_array(video_image))
                im_video = Image.fromarray(img_array_video)
                im_video.save('run'+str(RUN_NO)+'_video/frame_'+str(frame//5)+'.png')
                img_array = np.array(image_converter.to_rgb_array(main_image))
                im = Image.fromarray(img_array)
                # print(transform(im).shape)
                steer_to_save = int(100*cmd_steer*180./3.14)
                if frame//5 > 0 and not has_crossed_train_line(current_x,current_y):
                    im.save('run'+str(RUN_NO)+'_images/frame_'+str(frame//5)+'_'+str(steer_to_save)+\
                        '_'+str(int(10000*curvature_save))+'_'+str(int(100*x_save))+'_'+\
                        str(int(100*theta_save*180./3.14))+'_'+str(int(100*current_speed))+\
                        '_'+str(int(100*current_speed_perp))+'.png')
                # im = Image.open('run'+str(RUN_NO)+'_images/frame_'+str(frame//5)+'_'+str(steer_to_save)+'_'+str(int(10000*curvature_save))+'_'+str(int(100*x_save))+'_'+str(int(100*theta_save*180./3.14))+'.png')
                inp = torch.unsqueeze(transform(im),0).cuda()
                # print(inp.shape)
                preds = []
                mean = 0
                for i in range(N_ITERS) :
                    if VEHICLE_MODEL=='Kinematic' :
                        preds.append(float(model(inp)[0].cpu()))
                    else :
                        preds.append(float(model(inp,torch.tensor([[current_speed]]).cuda()\
                            ,torch.tensor([[current_speed_perp]]).cuda())[0].cpu()))
                    mean += preds[i]
                cmd_steer1 = mean/N_ITERS
                var = 0
                for pred in preds :
                    var += (pred-cmd_steer1)**2
                steer_var = (var/N_ITERS)**(1/2)*(3.14/180.)
                cmd_steer1 = cmd_steer1*(3.14/180.)
                Y2 = [0,0,0]
                
                preds = []
                mean = 0
                for i in range(N_ITERS) :
                    preds.append(float(model_safety_1(inp)[0].cpu()))
                    mean += preds[i]
                Y2[0] = mean/N_ITERS
                var = 0
                for pred in preds :
                    var += (pred-Y2[0])**2
                curvature_var = (var/N_ITERS)**(1/2)*curvature_factor

                preds = []
                mean = 0
                for i in range(N_ITERS) :
                    preds.append(float(model_safety_2(inp)[0].cpu()))
                    mean += preds[i]
                Y2[1] = mean/N_ITERS
                var = 0
                for pred in preds :
                    var += (pred-Y2[1])**2
                x_var = (var/N_ITERS)**(1/2)*x_factor
                
                preds = []
                mean = 0
                for i in range(N_ITERS) :
                    preds.append(float(model_safety_3(inp)[0].cpu()))
                    mean += preds[i]
                Y2[2] = mean/N_ITERS
                var = 0
                for pred in preds :
                    var += (pred-Y2[2])**2
                theta_var = (var/N_ITERS)**(1/2)*theta_factor*3.14/180.
                
                curvature = float(Y2[0])*curvature_factor
                x = float(Y2[1])*x_factor
                theta = float(Y2[2])*theta_factor*3.14/180.
                print("Before ", cmd_steer1, steer_var)
                print("Predicted theta, x, curvature : ", theta, x, x_var, curvature)
                print("Observed theta, x, curvature : ", theta_save, x_save, curvature_save)
                theta_comps.append([theta,theta_save])
                x_comps.append([x,x_var,x_save])
                curvature_comps.append([curvature,curvature_save])
            
                if measurement_data.player_measurements.collision_other > 0:
                    print("Collided")
                    break
            ###
            # Controller update (this uses the controller2d.py implementation)
            ###

            # To reduce the amount of waypoints sent to the controller,
            # provide a subset of waypoints that are within some 
            # a set of waypoints behind the car as well.
            
            # Find closest waypoint index to car. First increment the index
            # from the previous index until the new distance calculations
            # are increasing. Apply the same rule decrementing the index.
            # The final index should be the closest point (it is assumed that
            # the car will always break out of instability points where there
            # are two indices with the same minimum distance, as in the
            # center of a circle)
            closest_distance = np.linalg.norm(np.array([
                    waypoints_np[closest_index, 0] - current_x,
                    waypoints_np[closest_index, 1] - current_y]))
            new_distance = closest_distance
            new_index = closest_index
            while new_distance <= closest_distance:
                closest_distance = new_distance
                closest_index = new_index
                new_index += 1
                if new_index >= waypoints_np.shape[0]:  # End of path
                    break
                new_distance = np.linalg.norm(np.array([
                        waypoints_np[new_index, 0] - current_x,
                        waypoints_np[new_index, 1] - current_y]))
            new_distance = closest_distance
            new_index = closest_index
            while new_distance <= closest_distance:
                closest_distance = new_distance
                closest_index = new_index
                new_index -= 1
                if new_index < 0:  # Beginning of path
                    break
                new_distance = np.linalg.norm(np.array([
                        waypoints_np[new_index, 0] - current_x,
                        waypoints_np[new_index, 1] - current_y]))

            if closest_index >= waypoints_np.shape[0]-8:  # End of path
                break
            
            closest_distance = np.linalg.norm(np.array([
                    centerline_np[closest_index_centre, 0] - current_x,
                    centerline_np[closest_index_centre, 1] - current_y]))
            new_distance = closest_distance
            new_index = closest_index_centre
            while new_distance <= closest_distance:
                closest_distance = new_distance
                closest_index_centre = new_index
                new_index += 1
                if new_index >= centerline_np.shape[0]:  # End of path
                    break
                new_distance = np.linalg.norm(np.array([
                        centerline_np[new_index, 0] - current_x,
                        centerline_np[new_index, 1] - current_y]))
            new_distance = closest_distance
            new_index = closest_index_centre
            while new_distance <= closest_distance:
                closest_distance = new_distance
                closest_index_centre = new_index
                new_index -= 1
                if new_index < 0:  # Beginning of path
                    break
                new_distance = np.linalg.norm(np.array([
                        centerline_np[new_index, 0] - current_x,
                        centerline_np[new_index, 1] - current_y]))

            if closest_index_centre >= centerline_np.shape[0]-8:  # End of path
                break
            closest_angle_vector = centerline_np[closest_index_centre+5,:]-centerline_np[closest_index_centre,:]
            closest_angle = math.atan2(closest_angle_vector[1],closest_angle_vector[0])
            
            rel_angle = current_yaw - closest_angle
            
            # Once the closest index is found, return the path that has 1
            # waypoint behind and X waypoints ahead, where X is the index
            # that has a lookahead distance specified by 
            # INTERP_LOOKAHEAD_DISTANCE
            waypoint_subset_first_index = closest_index - 1
            if waypoint_subset_first_index < 0:
                waypoint_subset_first_index = 0

            waypoint_subset_last_index = closest_index
            total_distance_ahead = 0
            while total_distance_ahead < INTERP_LOOKAHEAD_DISTANCE:
                total_distance_ahead += wp_distance[waypoint_subset_last_index]
                waypoint_subset_last_index += 1
                if waypoint_subset_last_index >= waypoints_np.shape[0]:
                    waypoint_subset_last_index = waypoints_np.shape[0] - 1
                    break

            # Use the first and last waypoint subset indices into the hash
            # table to obtain the first and last indicies for the interpolated
            # list. Update the interpolated waypoints to the controller
            # for the next controller update.
            new_waypoints = \
                    wp_interp[wp_interp_hash[waypoint_subset_first_index]:\
                              wp_interp_hash[waypoint_subset_last_index] + 1]
            controller.update_waypoints(new_waypoints)

            # Update the other controller values and controls
            if VEHICLE_MODEL=='Kinematic' :
                controller.update_values(current_x, current_y, current_yaw, 
                                        current_speed,
                                        current_timestamp, frame)
            else :    
                current_yaw_updated = current_yaw+math.atan2(current_speed_perp,current_speed)/2.5
                controller.update_values(current_x, current_y, current_yaw_updated, 
                                        current_speed,
                                        current_timestamp, frame)
            controller.update_controls()
            cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
            vec1 = np.array([current_x,current_y])-centerline_np[closest_index_centre,:2]
            vec2 = np.array(closest_angle_vector)
            val = vec2[0]*vec1[1] - vec2[1]*vec1[0]
            x_save = closest_distance*val/abs(val)
            theta_save = rel_angle
            curvature_save = computeCurvature(centerline_np[closest_index_centre,:],\
                centerline_np[closest_index_centre+2,:],centerline_np[closest_index_centre+4,:])
            
            
            
            cmd_steer_updated = get_optimal_control(cmd_steer1, steer_var , current_speed, \
                theta_save, theta_var, x_save, x_var, curvature_save, curvature_var, \
                v_perp=current_speed_perp,omega=current_omega)
            # print("Updated : ", cmd_steer_updated)
            # Skip the first frame (so the controller has proper outputs)
            if skip_first_frame and frame == 0:
                pass
            else:
                # Update live plotter with new feedback
                # trajectory_fig.roll("trajectory", current_x, current_y)
                # trajectory_fig.roll("car", current_x, current_y)
                # When plotting lookahead path, only plot a number of points
                # (INTERP_MAX_POINTS_PLOT amount of points). This is meant
                # to decrease load when live plotting
                new_waypoints_np = np.array(new_waypoints)
                path_indices = np.floor(np.linspace(0, 
                                                    new_waypoints_np.shape[0]-1,
                                                    INTERP_MAX_POINTS_PLOT))
                # trajectory_fig.update("lookahead_path", 
                #         new_waypoints_np[path_indices.astype(int), 0],
                #         new_waypoints_np[path_indices.astype(int), 1],
                #         new_colour=[0, 0.7, 0.7])
                # forward_speed_fig.roll("forward_speed", 
                #                        current_timestamp, 
                #                        current_speed)
                # forward_speed_fig.roll("reference_signal", 
                #                        current_timestamp, 
                #                        controller._desired_speed)

                # throttle_fig.roll("throttle", current_timestamp, cmd_throttle)
                # brake_fig.roll("brake", current_timestamp, cmd_brake)
                # steer_fig.roll("steer", current_timestamp, cmd_steer1)

                # Refresh the live plot based on the refresh rate 
                # set by the options
                # if enable_live_plot and \
                #    live_plot_timer.has_exceeded_lap_period():
                #     lp_traj.refresh()
                #     lp_1d.refresh()
                #     live_plot_timer.lap()

            # Output controller command to CARLA server
            if RUN_NO == 0 or frame//5 < 0:
                send_control_command(client,
                                    throttle=cmd_throttle,
                                    steer=cmd_steer,
                                    brake=cmd_brake)
            else :
                send_control_command(client,
                                    throttle=cmd_throttle,
                                    steer=cmd_steer_updated,
                                    brake=cmd_brake)
            # Find if reached the end of waypoint. If the car is within
            # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
            # the simulation will end.
            dist_to_last_waypoint = np.linalg.norm(np.array([
                waypoints[-1][0] - current_x,
                waypoints[-1][1] - current_y]))
            if  dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT:
                reached_the_end = True
            if reached_the_end:
                break

        # End of demo - Stop vehicle and Store outputs to the controller output
        # directory.
        if reached_the_end:
            print("Reached the end of path. Writing to controller_output...")
        else:
            print("Exceeded assessment time. Writing to controller_output...")
        # Stop the car
        send_control_command(client, throttle=0.0, steer=0.0, brake=1.0)
        # Store the various outputs
        # store_trajectory_plot(trajectory_fig.fig, 'trajectory.png')
        # store_trajectory_plot(forward_speed_fig.fig, 'forward_speed.png')
        # store_trajectory_plot(throttle_fig.fig, 'throttle_output.png')
        # store_trajectory_plot(brake_fig.fig, 'brake_output.png')
        # store_trajectory_plot(steer_fig.fig, 'steer_output.png')
        write_trajectory_file(x_history, y_history, speed_history, time_history,steerings_list,throttles_list)
        theta_comps = np.array(theta_comps)
        x_comps = np.array(x_comps)
        curvature_comps = np.array(curvature_comps)
        np.savetxt('theta_comps.csv',theta_comps)
        np.savetxt('x_comps.csv',x_comps)
        np.savetxt('curvature_comps.csv',curvature_comps)

def main():
    global RUN_NO
    global model_path
    global N_ITERS
    global SAFEGUARD
    """Main function.

    Args:
        -v, --verbose: print debug information
        --host: IP of the host server (default: localhost)
        -p, --port: TCP port to listen to (default: 2000)
        -a, --autopilot: enable autopilot
        -q, --quality-level: graphics quality level [Low or Epic]
        -i, --images-to-disk: save images to disk
        -c, --carla-settings: Path to CarlaSettings.ini file
        -r, --run_no: Run no
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    argparser.add_argument(
        '-r', '--run_no',
        metavar='P',
        default=-1,
        type=int,
        help='Run no')
    args = argparser.parse_args()

    # Logging startup info
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    if args.run_no!=-1 :
        RUN_NO = args.run_no
    if RUN_NO < K_ITERS :
        SAFEGUARD = False
    if not SAFEGUARD :
        N_ITERS = 1
    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'
    model_path = 'saved_models_iter' + str(RUN_NO-1) 
    # Execute when server connection is established
    while True:
        try:
            exec_waypoint_nav_demo(args)
            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

