# Лифшиц. Механика сплошных сред


# зевс - 82 Гц (в юлижнем борисово - сверхдлинные волны, "голиаф")
# децимация, после фильтрации nu>80 Гц (ФНЧ)
# обрезать частоты nu<1/10 Гц (ФВЧ)

# Установка на севере - можем получить угол, под которым волна пришла с юга. 

# Грозовые центры -> глобальная электрическая цепь +грозы -> формирование шумновских резонансов

# 1) долина реки конго, 
# 2) долина реки амазонки
# 3) Полинезия

# Задача 1:
# цифровая обработка: оконная функция - перемножение с исх. сигналом 



# Задача 2:
# первый и последний участки H(t):
# 	отфильтровать (оставив 1 гармонику ш.р.)
# 	наложить спектры 1 гармоники, с разбиений участков и разделить на кол-во наложений 
# 	найти максимум - амплитуды на одной частоте резонанса

#   таким образом обработать два канала (СЮ, ЗВ) с одного датчика, найти отношение H_СЮ/H_ЗВ  >- arctg(phi), 
# 	phi - примерное положение источника 


# vlf.it (<22кГц)

import numpy as np
import matplotlib.pyplot as plt
# np.savetxt('test.out', np.array([x,y,z]).T)
# fig = plt.gcf()
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 8.5)
# x = np.arange(-5, 5, 1)
# y = np.arange(-5, 5, 1)
# xx, yy = np.meshgrid(x, y, sparse=True)
# z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
# plt.contourf(x,y,z)
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# name='x00'
# Generate data...
# x = np.random.random(100)
# y = np.random.random(10)
# z = x*y


data = np.genfromtxt('z2.out', skip_header=1, skip_footer=1).T
freq=data[0]
d1=data[1]
d2=data[2]
# d3=data[2]
# z = np.random.random(len(y))
# z=10*np.log10(z/0.2)
# z=np.abs((z))/np.max(z)-0.28
# i=np.where((z>0)&(y>0.1))
# z=z[i]
# x=x[i]
# y=y[i]
# print(z)
# plt.scatter(x, y, c=z, s=10, cmap='plasma')
# plt.plot(y,z)
# plt.title('Ch1')


# plt.subplot(1,1, 1)
ax.axvspan(7.6, 8.6, alpha=0.5, color='red')
ax.axvspan(13.10, 14.50, alpha=0.5, color='red')
ax.axvspan(19.30, 21.50, alpha=0.5, color='red')
ax.plot(freq, d2)

# plt.ylabel('Ch1 -- ЗенНад1')

# data = np.genfromtxt(name+'ch2.out', skip_header=1, skip_footer=1).T
# x=data[0]
# y=data[1]
# z=data[2]
# # z = np.random.random(len(y))
# z=10*np.log10(z/0.2)
# z=np.abs((z))/np.max(z)-0.28
# i=np.where((z>0)&(y<150)&(y>1))
# z=z[i]
# x=x[i]
# y=y[i]


# plt.subplot(3, 2, 3)
# plt.plot(y, z)
# plt.ylabel('Ch2 -- ВЗ1')

# data = np.genfromtxt(name+'ch3.out', skip_header=1, skip_footer=1).T
# x=data[0]
# y=data[1]
# z=data[2]
# # z = np.random.random(len(y))
# z=10*np.log10(z/0.2)
# z=np.abs((z))/np.max(z)-0.28
# i=np.where((z>0)&(y<150)&(y>1))
# z=z[i]
# x=x[i]
# y=y[i]


# plt.subplot(3, 2, 5)
# plt.plot(y, z)
# plt.ylabel('Ch3 -- CЮ1')

# data = np.genfromtxt(name+'ch4.out', skip_header=1, skip_footer=1).T
# x=data[0]
# y=data[1]
# z=data[2]
# # z = np.random.random(len(y))
# z=10*np.log10(z/0.2)
# z=np.abs((z))/np.max(z)-0.28
# i=np.where((z>0)&(y<150)&(y>1))
# z=z[i]
# x=x[i]
# y=y[i]


# plt.subplot(3, 2, 2)
# plt.plot(y, z)
# plt.ylabel('Ch4 -- ЗенНад2')

# data = np.genfromtxt(name+'ch5.out', skip_header=1, skip_footer=1).T
# x=data[0]
# y=data[1]
# z=data[2]
# # z = np.random.random(len(y))
# z=10*np.log10(z/0.2)
# z=np.abs((z))/np.max(z)-0.28
# i=np.where((z>0)&(y<150)&(y>1))
# z=z[i]
# x=x[i]
# y=y[i]


# plt.subplot(3, 2, 4)
# plt.plot(y, z)
# plt.ylabel('Ch5 -- ВЗ2')

# data = np.genfromtxt(name+'ch6.out', skip_header=1, skip_footer=1).T
# x=data[0]
# y=data[1]
# z=data[2]
# # z = np.random.random(len(y))
# z=10*np.log10(z/0.2)
# z=np.abs((z))/np.max(z)-0.28
# i=np.where((z>0)&(y<150)&(y>1))
# z=z[i]
# x=x[i]
# y=y[i]


# plt.subplot(3, 2, 6)
# plt.plot(y, z)
# plt.ylabel('Ch6 -- СЮ2')

# plt.title(name)
plt.show()