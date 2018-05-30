import numpy as np 

ch1=ch2=ch3=ch4=ch5=ch6=0
# for t in ['00','01','02']:
for t in ['57','58']:
	data = np.genfromtxt('res'+t, skip_header=1, skip_footer=0).T
	arr=data[1:].T
	ch1+=arr[0]
	ch2+=arr[1]
	ch3+=arr[2]
	ch4+=arr[3]
	ch5+=arr[4]
	ch6+=arr[5]
ch1=ch1/3
ch2=ch2/3
ch3=ch3/3
ch4=ch4/3
ch5=ch5/3
ch6=ch6/3
# print(ch1)
# print(ch2)
# print(ch3)
# print(ch4)
# print(ch5)
# print(ch6)

alpha2=np.arctan(ch2[1]/ch3[1])/np.pi*180
alpha50=np.arctan(ch2[2]/ch3[2])/np.pi*180
alpha82=np.arctan(ch2[3]/ch3[3])/np.pi*180
print(alpha2,alpha50,alpha82)
# freq=data[0]
# d1=data[1]
# d2=data[2]