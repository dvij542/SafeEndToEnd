import numpy as np
import math

n_iters = 10
parent_dir = './cent_line_dynamic_with_cbf/controller_output/'
ref_line = np.loadtxt(parent_dir + 'trajectory_run'+str(0)+'.txt',delimiter=',')
dt = 0.1

def find_min_dist(p) :
    x,y = p[0], p[1]
    dists = (ref_line[:,:2]-np.array([[x,y]]))
    dist = dists[:,0]**2 + dists[:,1]**2
    # print(min(dist))
    mini = np.argmin(dist)
    vals = []
    if mini>0 :
        x1,y1 = ref_line[mini-1,0],ref_line[mini-1,1]
        x2,y2 = ref_line[mini,0],ref_line[mini,1]
        a,b,c = -(y2-y1), (x2-x1),y2*x1-y1*x2 
        vals.append(abs((a*x+b*y+c)/(math.sqrt(a**2+b**2))))
    if mini < len(ref_line)-1 :
        x1,y1 = ref_line[mini,0],ref_line[mini,1]
        x2,y2 = ref_line[mini+1,0],ref_line[mini+1,1]
        a,b,c = -(y2-y1), (x2-x1),y2*x1-y1*x2 
        vals.append(abs((a*x+b*y+c)/(math.sqrt(a**2+b**2))))
    # print("aa : ", min(vals))
    return min(vals)

for i in range(n_iters+1) :
    fname = parent_dir + 'trajectory_run'+str(i)+'.txt'
    traj = np.loadtxt(fname,delimiter=',')[:,:]
    mean_dev = 0
    for j in range(20,len(traj)) :
        dist = find_min_dist(traj[j,:2])
        mean_dev += dist
    # print(mean_dev)
    print("Traj ", str(i), " :-")
    print("Mean deviation : ", mean_dev/(len(traj)-20))
    print("Total time : ", len(traj)*dt)
