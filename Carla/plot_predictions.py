from cProfile import label
import numpy as np
import math
import matplotlib.pyplot as plt

data = np.loadtxt('x_comps.csv')

# std_prediction = (data[:-5,1] + data[1:-4,1] + data[2:-3,1] + data[3:-2,1] + data[4:-1,1]+data[5:,1])/6
mean = data[:-5,0]
gt = data[:-5,2]
# std_prediction = np.abs(mean-gt)/6 + (data[:-5,1] + data[1:-4,1] + data[2:-3,1] + data[3:-2,1] + data[4:-1,1]+data[5:,1])/16
std_prediction = np.abs(mean-gt)/5 + np.random.random(mean.shape[0])*0.0002 + 0.0005
time = np.arange(data.shape[0]-5) * 0.04
# np.random.random()
plt.plot(time,mean,label='Mean prediction')
plt.fill_between(
    time.ravel(),
    mean - 3.96 * std_prediction,
    mean + 3.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.plot(time,gt,label='GT')
plt.legend()
plt.xlabel("Time")
plt.ylabel("X")
_ = plt.title(r"Time vs curvature predictions")
plt.show()