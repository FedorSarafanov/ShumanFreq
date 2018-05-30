#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy import signal
from numpy.fft import rfft, rfftfreq
from numpy.random import uniform
import sys

# name=sys.argv[1]
# print(name)
name='x57'
data = np.genfromtxt('_'+name, skip_header=1, skip_footer=1).T

print(name)
t=data[0]

ch1=data[1]
ch2=data[2]
ch3=data[3]
ch4=data[4]
ch5=data[5]
ch6=data[6]

t=t/1000

N=len(t)
T=t[10]-t[9]
x=t
# noise = uniform(-200.,200., N)

j=0
for ch in [ch3]:
	j+=1
	b, a = signal.butter(8, 0.007, btype='highpass')
	ch = scipy.signal.filtfilt(b, a, ch, padlen=150)
	b, a = signal.butter(8, 0.125)
	ch = scipy.signal.filtfilt(b, a, ch, padlen=150)
	# highpass
	# ch4 = scipy.signal.filtfilt(ch4, 1.0, x)
	# amp = np.abs(rfft(y))/N*2
	# print('Старт вычислений -- ch',j)
	amp1 = np.abs(rfft(ch))/N*2
	# print('Амплитуды рассчитаны')
	freq=rfftfreq(N, T)

	i=0
	# ind=0
	nd=100
	for f in freq:
		if np.mod(i,nd)==0:
			# indi=np.linspace(i-nd,i, dtype=np.int16)
			indi=np.arange(i-nd,i)
			aar=np.sum(amp1[indi])/nd
			amp1[indi]=aar
		i+=1

	A=amp1

	g8i=np.where((freq>6.6)&(freq<7.14))
	g14i=np.where((freq>13.2)&(freq<14.4))
	g50i=np.where((freq>49.4)&(freq<50.8))
	g82i=np.where((freq>80.7)&(freq<82.1))

	g8=np.median(A[g8i])
	g14=np.median(A[g14i])
	g50=np.median(A[g50i])
	g82=np.median(A[g82i])

	print('ch'+str(j),g8,g14,g50,g82)

# tau=np.ones(len(freq))*t[0]
# np.savetxt('z2.out', np.array([freq,np.abs(amp1),np.abs(amp2),np.abs(amp3)]).T)
# print('ch1')
# amp2 = np.abs(rfft(ch2+noise))/N*2
# np.savetxt(name+'ch2.out', np.array([tau,freq,np.abs(amp2)]).T)
# print('ch2')
# amp3 = np.abs(rfft(ch3+noise))/N*2
# np.savetxt(name+'ch3.out', np.array([tau,freq,np.abs(amp3)]).T)
# print('ch3')
# amp4 = np.abs(rfft(ch4+noise))/N*2
# np.savetxt(name+'ch4.out', np.array([tau,freq,np.abs(amp4)]).T)
# print('ch4')
# amp5 = np.abs(rfft(ch5+noise))/N*2
# np.savetxt(name+'ch5.out', np.array([tau,freq,np.abs(amp5)]).T)
# print('ch5')
# amp6 = np.abs(rfft(ch6+noise))/N*2
# np.savetxt(name+'ch6.out', np.array([tau,freq,np.abs(amp6)]).T)
# print('ch6')

# xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

# fig, ax1 = plt.subplots()
fig, (ax1, ax2) = plt.subplots(1, 2)
# ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
ax1.plot(freq,amp1/34)
ax1.fill_between(freq, 0, amp1/28)

x=np.linspace(0,1,1000);
y=1/(1+(x/0.5)**6);
ax2.plot(x,y);
# ax.hist([1, 2, 1], bins=[0, 1, 2, 3])
# plt.axis([np.min(freq)+0.1, np.max(freq),0,100])
# print(len(tau),len(amp1))
# ax.plot(freq,np.log(np.abs(amp1)))
# ax.axvspan(7.6, 8.6, alpha=0.5, color='red')
# ax.axvspan(13.10, 14.50, alpha=0.5, color='red')
# ax.axvspan(19.30, 21.50, alpha=0.5, color='red')
# plt.axvline(x=7.83,color='r')
# plt.axvline(x=14.1,color='r')
# plt.axvline(x=20.3,color='r')
plt.show()