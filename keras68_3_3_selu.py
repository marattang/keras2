import numpy as np
import matplotlib.pyplot as plt

def selu(x, alpha):

    return (x>0)*x + (x<=0)+(alpha+(np.exp(x)-1))

x = np.arange(-5, 5, 0.1)
y = selu(x, 2)

plt.plot(x, y)
plt.grid()
plt.show()

# 과제
# elu, selu, reaky relu 
# 68_3_2, 3, 4로 만들기
