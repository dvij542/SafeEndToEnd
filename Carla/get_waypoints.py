import numpy as np

waypoints = np.loadtxt('racetrack_waypoints.txt',delimiter=',')
track = np.loadtxt('traj_race_cl.csv',delimiter=';')
waypoints_out = np.zeros((track.shape[0],3))

i=0
for point in track :
    x,y = point[1], point[2]
    waypoints_out[i][0] = x
    waypoints_out[i][1] = y
    dists = np.sum((waypoints[:,:2] - np.array([[x,y]]))**2,axis=1)
    waypoints_out[i][2] = waypoints[np.argmin(dists)][2]
    i += 1

np.savetxt('waypoints_new.csv',waypoints_out,delimiter=',')