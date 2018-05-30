#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from numpy.fft import rfft, rfftfreq
from numpy.random import uniform
import sys

# name=sys.argv[1]
# print(name)
name='x00'
data = np.genfromtxt('_'+name, skip_header=1, skip_footer=1).T

t=data[0]

ch1=data[1]
ch2=data[2]
ch3=data[3]
ch4=data[4]
ch5=data[5]
ch6=data[6]

N = 50000
T = 0.0005

x = np.linspace(0.0, N*T, N)
y = 100*np.sin(1 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)+0.5*np.sin(90.0 * 2.0*np.pi*x)+10*np.sin(400.0 * 2.0*np.pi*x)
# yf = scipy.fftpack.fft(y)

t=t/1000

N=len(t)
T=t[10]-t[9]
x=t
# noise = uniform(-200.,200., N)
# y=ch1

from scipy import signal
f1,f2 = 0.1,150
# f1,f2 = 0.2,1
# f1,f2 = 7.6,8.6
# f3,f4 = 13.10, 14.50
# f5, f6 =19.30, 25
# arr=signal.firwin(N, f)
arr=signal.firwin(N, [f1, f2], pass_zero=False, fs=2000,window='blackman')

b, a = signal.butter(8, 0.007, btype='highpass')
ch4 = scipy.signal.filtfilt(b, a, ch4, padlen=150)
b, a = signal.butter(8, 0.125)
ch4 = scipy.signal.filtfilt(b, a, ch4, padlen=150)
# highpass
# ch4 = scipy.signal.filtfilt(ch4, 1.0, x)
# amp = np.abs(rfft(y))/N*2
print('Старт вычислений')
amp1 = np.abs(rfft(ch4))/N*2
print('Амплитуды рассчитаны')
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

print(g8,g14,g50,g82)

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

fig, ax = plt.subplots()
# ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
ax.plot(freq,amp1)


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