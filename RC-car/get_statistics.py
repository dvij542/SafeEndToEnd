import numpy as np
import math

n_iters = 9
parent_dir = './controller_output_with_cbf/'
ref_line = np.loadtxt('raceline3.csv')
dt = 0.15

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

for i in range(0,n_iters+1) :
    fname = parent_dir + 'iter'+str(i)+'_0.csv'
    traj = np.loadtxt(fname,delimiter=',')
    mean_dev = 0
    for j in range(len(traj)) :
        dist = find_min_dist(traj[j,:2])
        mean_dev += dist
    print("Traj ", str(i), " :-")
    print("Mean deviation : ", mean_dev/len(traj))
    print("Total time : ", traj[-1,-1]-traj[0,-1])
