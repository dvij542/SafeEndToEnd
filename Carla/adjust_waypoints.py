import numpy as np
import matplotlib.pyplot as plt
import math
file_csv = np.loadtxt('waypoints_2.csv',delimiter=',')
# file_csv1 = np.loadtxt('ttl.csv',delimiter=',')
theta = 6*(3.14/180.)
x,y = file_csv[:,0]*math.cos(theta)-file_csv[:,1]*math.sin(theta),file_csv[:,1]*math.cos(theta)+file_csv[:,0]*math.sin(theta) 
plt.plot(file_csv[:,0],file_csv[:,1])
# plt.plot(file_csv1[:,0],file_csv1[:,1])
plt.plot(x,y)
new_file_csv = np.array([x,y]).T
np.savetxt('centre_line_2.csv',new_file_csv,delimiter=',')
plt.show()
