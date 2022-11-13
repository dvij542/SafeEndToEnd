import numpy as np

waypoints = np.loadtxt('racetrack_waypoints.txt',delimiter=',')
track = np.zeros((waypoints.shape[0],4))

i=0
for point in waypoints :
    x,y = point[0], point[1]
    track[i][0] = x
    track[i][1] = y
    track[i][2] = 6
    track[i][3] = 6
    i += 1

np.savetxt('track.csv',track,delimiter=',',header='# x_m,y_m,w_tr_right_m,w_tr_left_m')