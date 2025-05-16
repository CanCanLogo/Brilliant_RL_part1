#输入数据和标签
x_data = [338.,333.,328.,207.,226.,25.,179.,60.,208.,606.]
y_data = [640.,633.,619.,393.,428.,27.,193.,66.,226.,1591]

#初始值和超参数设置
b = -150
w = 0
# lr = 0.0000001
lr = 0.0000001
iteration = 100000
leng = len(x_data)
print(leng)

#w_list和b_list用于记录每一轮迭代的w和b值，用于绘图
w_list = [float]*iteration
b_list = [float]*iteration
for i in range(iteration):
    b_grad = 0
    w_grad = 0
    #填空部分，实现梯度下降算法
    if i != 0:
        w = w_list[i-1]
        b = b_list[i-1]
    y_pred = [i * w + b for i in y_data]
    for j in range(leng):
        w_grad += (y_pred[j]-y_data[j])*x_data[j]
        b_grad += (y_pred[j]-y_data[j])
    w = w - lr * w_grad
    b = b - lr * b_grad
    w_list[i] = w
    b_list[i] = b
print(w)
print(b)
#绘图部分
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# plt.plot(x_data, y_data, 'ro')
_X = [0, 600]
_Y = [b + w * x for x in _X]

plt.plot(x_data, y_data, 'ro', _X, _Y, 'b', linewidth=2)
# plt.plot( _X, _Y, 'b', linewidth=2)
# plt.title("y = {} + {}x".format(a0, a1))
plt.show()



'''
fig = plt.figure()
plt.xlim(-200,-80)
plt.ylim(-4,4)

#设置背景
xmin, xmax = xlim = -200,-80
ymin, ymax = ylim = -4,4
ax = fig.add_subplot(111, xlim=xlim, ylim=ylim,
                     autoscale_on=False)
X = [[4, 4],[4, 4],[4, 4],[1, 1]]
ax.imshow(X, interpolation='bicubic', cmap=cm.Spectral,
          extent=(xmin, xmax, ymin, ymax), alpha=1)
ax.set_aspect('auto')

#绘制每一个数据点
plt.scatter(b_list,w_list,s=2,c='black',label=(lr,iteration))
plt.title("lr:0.000001 iter:500000")
plt.legend()
plt.show()
'''


