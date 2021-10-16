import numpy as np
import matplotlib.pyplot as plt 

qlearning = np.load('qlearning.npy')
sarsa = np.load('sarsa.npy')

# for i in range(500):
#     while qlearning[i]<-100:
#         qlearning[i] = np.nan
#     while sarsa[i]<-100:
#         sarsa[i] += np.nan

x = np.arange(1,501,1)
plt.plot(x,qlearning)
plt.plot(x,sarsa)

plt.xlabel('episodes')
plt.ylabel('sum of reward')

plt.xlim(0,500)
plt.ylim(-100,0)

plt.show()