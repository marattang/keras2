# 난 정말 시그모이드~

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5, 5, 0.1)
print(len(x))
print(x.shape)

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()

# 이전에 진공관을 사용할 시적에는
# 0아니면 1밖에 없는 계층함수를 썼다
