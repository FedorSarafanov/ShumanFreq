import numpy as np
# from numpy import sin,cos,pi
# import matplotlib.pyplot as plt

data = np.genfromtxt('_x00', skip_header=1, skip_footer=1).T

t=data[0]

ch1=data[1]
ch2=data[2]
ch3=data[3]
ch4=data[4]
ch5=data[5]
ch6=data[6]

# plt.plot(t,ch1)
# plt.show()
