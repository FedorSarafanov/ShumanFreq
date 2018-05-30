import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
x=[1,2,3]
y=[1,5,3]
z=[2,8,5]
# np.savetxt('test.out', np.array([x,y,z]).T)
# fig = plt.gcf()
# fig.set_size_inches(180.5, 8.5)
# x = np.arange(-5, 5, 1)
# y = np.arange(-5, 5, 1)
# xx, yy = np.meshgrid(x, y, sparse=True)
# z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
# plt.contourf(x,y,z)
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
name='x00'
# Generate data...
# x = np.random.random(100)
# y = np.random.random(10)
# z = x*y

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')


data = np.genfromtxt(name+'ch1.out', skip_header=1, skip_footer=1).T
x=data[0]

y=data[1]
z=data[2]
# z = np.random.random(len(y))
z=10*np.log10(z/0.2)
z=np.abs((z))/np.max(z)-0.28
i=np.where((z>0)&(y<150)&(y>1))
z=z[i]
x=x[i]
y=y[i]
# print(z)
# plt.scatter(x, y, c=z, s=10, cmap='plasma')
# plt.plot(y,z)
# plt.title('Ch1')


# plt.subplot(3, 2, 1)
ax1.plot(y, z)
# ax.ylabel('Ch1 -- ЗенНад1')

data = np.genfromtxt(name+'ch2.out', skip_header=1, skip_footer=1).T
x=data[0]
y=data[1]
z=data[2]
# z = np.random.random(len(y))
z=10*np.log10(z/0.2)
z=np.abs((z))/np.max(z)-0.28
i=np.where((z>0)&(y<150)&(y>1))
z=z[i]
x=x[i]
y=y[i]


# plt.subplot(3, 2, 3)
ax2.plot(y, z)
# plt.ylabel('Ch2 -- ВЗ1')

data = np.genfromtxt(name+'ch3.out', skip_header=1, skip_footer=1).T
x=data[0]
y=data[1]
z=data[2]
# z = np.random.random(len(y))
z=10*np.log10(z/0.2)
z=np.abs((z))/np.max(z)-0.28
i=np.where((z>0)&(y<150)&(y>1))
z=z[i]
x=x[i]
y=y[i]


# plt.subplot(3, 2, 5)
ax3.plot(y, z)
# plt.ylabel('Ch3 -- CЮ1')

data = np.genfromtxt(name+'ch4.out', skip_header=1, skip_footer=1).T
x=data[0]
y=data[1]
z=data[2]
# z = np.random.random(len(y))
z=10*np.log10(z/0.2)
z=np.abs((z))/np.max(z)-0.28
i=np.where((z>0)&(y<150)&(y>1))
z=z[i]
x=x[i]
y=y[i]


# plt.subplot(3, 2, 2)
ax4.plot(y, z)
# plt.ylabel('Ch4 -- ЗенНад2')

data = np.genfromtxt(name+'ch5.out', skip_header=1, skip_footer=1).T
x=data[0]
y=data[1]
z=data[2]
# z = np.random.random(len(y))
z=10*np.log10(z/0.2)
z=np.abs((z))/np.max(z)-0.28
i=np.where((z>0)&(y<150)&(y>1))
z=z[i]
x=x[i]
y=y[i]


# plt.subplot(3, 2, 4)
ax5.plot(y, z)
# plt.ylabel('Ch5 -- ВЗ2')

data = np.genfromtxt(name+'ch6.out', skip_header=1, skip_footer=1).T
x=data[0]
y=data[1]
z=data[2]
# z = np.random.random(len(y))
z=10*np.log10(z/0.2)
z=np.abs((z))/np.max(z)-0.28
i=np.where((z>0)&(y<150)&(y>1))
z=z[i]
x=x[i]
y=y[i]


# plt.subplot(3, 2, 6)
ax6.plot(y, z)
# plt.ylabel('Ch6 -- СЮ2')

ax1.set_title('Канал 1, 1 вертикальный')
ax2.set_title('Канал 2, 1 запад-восток')
ax3.set_title('Канал 3, 1 север-юг')
ax4.set_title('Канал 4, 2 вертикальный')
ax5.set_title('Канал 5, 2 запад-восток')
ax6.set_title('Канал 6, 2 север-юг')
# plt.title(name)
plt.show()
