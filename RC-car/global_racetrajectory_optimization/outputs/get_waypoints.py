import numpy as np

# waypoints = np.loadtxt('traj_3.csv',delimiter=',')
# track = np.loadtxt('traj_race_cl.csv',delimiter=';')
# waypoints_out = np.zeros((track.shape[0],3))

# i=0
# for point in track :
#     x,y = point[1], point[2]
#     waypoints_out[i][0] = x
#     waypoints_out[i][1] = y
#     dists = np.sum((waypoints[:,:2] - np.array([[x,y]]))**2,axis=1)
#     waypoints_out[i][2] = 1
#     i += 1

# np.savetxt('raceline1.csv',waypoints_out,delimiter=',')

import matplotlib.pyplot as plt
# track = np.loadtxt('traj_race_cl.csv',delimiter=';')
center_line = np.loadtxt('path_followed.csv',delimiter=',')
# race_line1 = np.loadtxt('raceline1.csv',delimiter=',')
race_line2 = np.loadtxt('raceline.csv',delimiter=',')
plt.plot(center_line[:,0],center_line[:,1])
# plt.plot(race_line1[:,0],race_line1[:,1])
plt.plot(race_line2[:,0],race_line2[:,1])
plt.axis('equal')
plt.show()