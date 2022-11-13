import matplotlib.pyplot as plt
import numpy as np
import math

p = [[10,10],[15,15],[30,30]]

for pi in p :
    rhos = []
    x = pi[0]
    y = pi[1]
    thetas = np.arange(0.,180.,1.)
    for theta in thetas :
        rho = x*math.cos(theta*math.pi/180.) + y*math.sin(theta*math.pi/180.)
        rhos.append([theta,rho])
    rhos = np.array(rhos)
    plt.plot(rhos[:,0],rhos[:,1],label='For point (' + str(x) + ',' + str(y) + ')')

plt.xlabel(r'$\theta$ (in $\degree$)')
plt.ylabel(r'$\rho$')
plt.legend()
plt.show()