import numpy as np
# import matplotlib.pyplot as plt
# import scipy.fftpack
freq=np.linspace(0,100)
amp1=np.linspace(0,100)
i=0
# ind=0
nd=4
print(freq)
for f in freq:
	if (np.mod(i,nd)==0)&(i!=0):
		indi=np.arange(i-nd,i)
		print(indi)
		aar=np.sum(amp1[indi])/nd
		amp1[indi]=aar
	i+=1
print(amp1)