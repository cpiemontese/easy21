from monte_carlo import MonteCarloAgent
from sarsa_lambda import SarsaLambdaAgent

import numpy as np
import matplotlib.pyplot as plt 

mca = MonteCarloAgent(EPS_CONST=10000.0)
mca.learn(episodes=1000000)
Q_star = mca.get_Q()        # Q* for mean squares comparison
#mca.plot_value_fun()

lambdas = np.linspace(0.0, 1.0, num=11)

slam = SarsaLambdaAgent()

#mse = [0.0] * 11 
m = 0

mse_per_episode = [0.0] * 1000

#for l in lambdas:
    #print('lambda {}'.format(l))
slam.set_lambda(1.0)
for i in range(1000):
    slam.learn(episodes=1)
    Q = slam.get_Q()
    for i in range(10):
        for j in range(21):
            for a in ['stick', 'hit']:
                mse_per_episode[m] += pow(Q[i][j][a] - Q_star[i][j][a], 2.0)
    m += 1

    '''for i in range(10):
        for j in range(21):
            for a in ['stick', 'hit']:
                mse[m] += pow(Q[i][j][a] - Q_star[i][j][a], 2.0)
    m += 1'''
    #slam.plot_value_fun()

plt.plot(mse_per_episode)
plt.xlabel('episodes')
plt.ylabel('mse')
plt.title('lambda 1.0')
plt.show()

#plt.plot(lambdas, mse)
#plt.xlabel('lambda')
#plt.ylabel('mse')
#plt.show()