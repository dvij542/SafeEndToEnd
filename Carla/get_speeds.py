import numpy as np
import math 

racing_line = np.loadtxt('traj_race_cl.csv',delimiter=';')[:,[1,2,3]]
center_line = np.loadtxt('racetrack_waypoints.txt',delimiter=',')

def dist(x1,y1,x2,y2) :
    return (x1-x2)**2 + (y1-y2)**2
i = 0
for row in racing_line :
    x,y = row[0],row[1]
    minval = dist(x,y,center_line[0,0],center_line[0,1])
    t = 0
    mint = 0
    for row1 in center_line :
        x1,y1 = row1[0], row1[1]
        d = dist(x,y,x1,y1)
        if d < minval :
            minval = d
            mint = t
        t += 1 
    racing_line[i,2] = center_line[mint,2]
    i+=1 

np.savetxt('racing_line.txt',racing_line,delimiter=',')