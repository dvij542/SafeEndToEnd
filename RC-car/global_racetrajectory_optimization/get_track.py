import numpy as np

LANE_WIDTH = 0.55
waypoints = np.loadtxt('../centre_line.csv',delimiter=',')
track = np.zeros((waypoints.shape[0],4))

i=0
for point in waypoints :
    x,y = point[0], point[1]
    track[i][0] = x
    track[i][1] = y
    track[i][2] = LANE_WIDTH
    track[i][3] = LANE_WIDTH
    i += 1

np.savetxt('track1.csv',track,delimiter=',',header='# x_m,y_m,w_tr_right_m,w_tr_left_m')