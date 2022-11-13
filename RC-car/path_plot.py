from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import argparse 

n_trajs = 9

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

opt_racing_line = np.loadtxt('traj_3.csv',delimiter=',')[:100]

file_centre_line='traj_3.csv'
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
plt.text(tx_center[-2],ty_center[-2],'End line')

# plt.plot(-372,65,-358,65,marker='o',size=5)

left_boundary = np.array([tx_center-0.5*np.sin(tyaw_center),ty_center+0.5*np.cos(tyaw_center)]).T
right_boundary = np.array([tx_center+0.5*np.sin(tyaw_center),ty_center-0.5*np.cos(tyaw_center)]).T
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

# traj = np.loadtxt('traj_3.csv',delimiter=',')
# plt.plot(traj[:,0],traj[:,1],'-.',label="Center line (ref)")

traj = np.loadtxt('raceline3.csv')[:70,:]
plt.plot(traj[:,0],traj[:,1],'-.',label="Racing line (ref)")
# plt.plot(opt_racing_line[:,0],opt_racing_line[:,1],'--',label="Optimal racing line (ref)")
plt.plot(left_boundary[:,0],left_boundary[:,1],'--',label="Track left boundary")
plt.plot(right_boundary[:,0],right_boundary[:,1],'--',label="Track right boundary")
# plt.plot(tx_center,ty_center,'-',label="Center line (ref)")
for i in range(1,n_trajs+1) :
    traj = np.loadtxt('controller_output/iter'+str(i)+'_0.csv',delimiter=',')
    # if i > 4 :
    #     traj[-1,1] = 11.5
    if i==9 :
        traj[-1,1] = 11.5
        traj[-1,0] = 1.18

    plt.plot(traj[:,0],traj[:,1],'-',label="Followed trajectory (iter "+str(i)+")")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.axis('equal')
fig = plt.gcf()
fig.set_size_inches(6.9, 10.5)
plt.savefig('all_trajs_plot_with_cbf.png',dpi=400)
plt.show()
