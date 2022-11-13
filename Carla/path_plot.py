from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import argparse 

n_trajs = 11

argparser = argparse.ArgumentParser()
argparser.add_argument(
        '-n', '--run_no',
        metavar='P',
        default=-1,
        type=int,
        help='Run no')
args = argparser.parse_args()

if args.run_no != -1 :
    n_trajs = args.run_no

opt_racing_line = np.loadtxt('waypoints_new.csv',delimiter=',')[:100]

file_centre_line='racetrack_waypoints.txt'
if file_centre_line != None:
    centre_line = np.loadtxt(file_centre_line,delimiter = ",")
else :
    centre_line=None
# centre_line[:,1] = -centre_line[:,1]
tx_center, ty_center, tyaw_center = centre_line[:-1,0], centre_line[:-1,1], np.arctan2(centre_line[1:,1]-centre_line[:-1,1],centre_line[1:,0]-centre_line[:-1,0])

# Start line
plt.plot([tx_center[0]+np.cos(tyaw_center[0]+math.pi/2),tx_center[0]-np.cos(tyaw_center[0]+math.pi/2)],[ty_center[0]+np.sin(tyaw_center[0]+math.pi/2),ty_center[0]-np.sin(tyaw_center[0]+math.pi/2)],linewidth=5.0,color='green')#,marker='o')
plt.text(tx_center[0],ty_center[0],'Start line')

# Finish line
plt.plot([tx_center[-1]+np.cos(tyaw_center[-1]+math.pi/2),tx_center[-1]-np.cos(tyaw_center[-1]+math.pi/2)],[ty_center[-1]+np.sin(tyaw_center[-1]+math.pi/2),ty_center[-1]-np.sin(tyaw_center[-1]+math.pi/2)],linewidth=5.0,color='red')#,marker='o')
plt.text(tx_center[-1],ty_center[-1],'End line')

# plt.plot(-372,65,-358,65,marker='o',size=5)

left_boundary = np.array([tx_center-7*np.sin(tyaw_center),ty_center+7*np.cos(tyaw_center)]).T
right_boundary = np.array([tx_center+7*np.sin(tyaw_center),ty_center-7*np.cos(tyaw_center)]).T
# traj1 = np.loadtxt('controller_output/trajectory_run1.txt',delimiter=',')
# traj2 = np.loadtxt('controller_output/trajectory_run2.txt',delimiter=',')
# traj3 = np.loadtxt('controller_output/trajectory_run3.txt',delimiter=',')
# traj4 = np.loadtxt('controller_output/trajectory_run4.txt',delimiter=',')
# traj5 = np.loadtxt('controller_output/trajectory_run5.txt',delimiter=',')
# traj7 = np.loadtxt('controller_output/trajectory_run7.txt',delimiter=',')
# plt.plot(traj1[:,0],traj1[:,1],'-',label="Followed trajectory (iter 1)")
# plt.plot(traj2[:,0],traj2[:,1],'-',label="Followed trajectory (iter 2)")
# plt.plot(traj3[:,0],traj3[:,1],'-',label="Followed trajectory (iter 3)")
# plt.plot(traj4[:,0],traj4[:,1],'-',label="Followed trajectory (iter 4)")
# plt.plot(traj5[:,0],traj5[:,1],'-',label="Followed trajectory (iter 5)")

traj = np.loadtxt('with_cbf_dynamic_updated/controller_output/trajectory_run0.txt',delimiter=',')
plt.plot(traj[:,0],traj[:,1],'-.',label="Center line (ref)")

# plt.plot(opt_racing_line[:,0],opt_racing_line[:,1],'--',label="Optimal racing line (ref)")
plt.plot(left_boundary[:,0],left_boundary[:,1],'--',label="Track left boundary")
plt.plot(right_boundary[:,0],right_boundary[:,1],'--',label="Track right boundary")
# plt.plot(tx_center,ty_center,'-',label="Center line (ref)")
for i in range(1,n_trajs+1) :
    traj = np.loadtxt('with_cbf_dynamic_updated/controller_output/trajectory_run'+str(i)+'.txt',delimiter=',')
    plt.plot(traj[:,0],traj[:,1],'-',label="Followed trajectory (iter "+str(i)+")")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.axis('equal')
fig = plt.gcf()
fig.set_size_inches(6.9, 10.5)
plt.savefig('all_trajs_plot.png',dpi=400)
plt.show()
