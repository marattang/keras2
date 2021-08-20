import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 -4*x + 6

gradient = lambda x: 2*x - 4
# 최적의 weight, loss자리 = gradient가 0이 되는 자리?
def learning_rate(lr):
    x0 = 0.0
    MaxIter = 20
    learning_rate = lr

    print('step\tx\tf(x)')
    print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))
    x1_value = [x0]
    x0_value = [x0]
    epochs = []
    idx = 0
    for i in range(MaxIter):
        x1 = x0 - learning_rate * gradient(x0) #미분함수(gradient에 x값을 넣음)
        x0 = x1
        x1_value.append(x0)
        x0_value.append(f(x0))
        if f(x0) == x0:
            idx+=1
            epochs.append(i)
        print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0)))

    return {'x1_value':x1_value, 'x0_value':x0_value, 'epochs':epochs, 'idx':idx}

x = np.linspace(-1, 6, 100)
print(x)
y = f(x)

# print('keys', lr25['x1_value'])
lr = [0.25, 0.5, 0.75, 1.0]
for i, j in enumerate(lr):
    lrl = learning_rate(j)
    plt.subplot(f'{220+i}')
    plt.plot(x, y, 'k-')
    plt.plot(lrl['x1_value'], lrl['x0_value'], 'k-', color='red')
    plt.plot(2, 2, 'sk')
    plt.title(f'learnign rate : {j} epchos:{lrl["epochs"]} idx:{lrl["idx"]}')
    plt.grid()
    plt.xlabel('x')
    plt.xlabel('y')
plt.show()

'''
learning rate = 0.8     =>      learning rate = 0.25
step    x       f(x)            step    x       f(x)
00      0.00000 6.00000         00      0.00000 6.00000      
01      3.20000 3.44000         01      1.00000 3.00000 
02      1.28000 2.51840         02      1.50000 2.25000 
03      2.43200 2.18662         03      1.75000 2.06250 
04      1.74080 2.06718         04      1.87500 2.01562 
05      2.15552 2.02419         05      1.93750 2.00391 
06      1.90669 2.00871         06      1.96875 2.00098 
07      2.05599 2.00313         07      1.98438 2.00024 
08      1.96641 2.00113         08      1.99219 2.00006 
09      2.02016 2.00041         09      1.99609 2.00002 
10      1.98791 2.00015         10      1.99805 2.00000 
11      2.00726 2.00005         11      1.99902 2.00000 
12      1.99565 2.00002         12      1.99951 2.00000 
13      2.00261 2.00001         13      1.99976 2.00000 
14      1.99843 2.00000         14      1.99988 2.00000 
15      2.00094 2.00000         15      1.99994 2.00000 
16      1.99944 2.00000         16      1.99997 2.00000 
17      2.00034 2.00000         17      1.99998 2.00000 
18      1.99980 2.00000         18      1.99999 2.00000 
19      2.00012 2.00000         19      2.00000 2.00000 
20      1.99993 2.00000         20      2.00000 2.00000 
'''