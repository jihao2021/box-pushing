import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

#small_box_success_rate = [0.33, 0.67, 0.74, 0.91, 0.98, 0.98, 0.97, 0.97, 0.97, 0.98 ]
#x = small_box_success_rate
#plt.bar(x, height=1.0)
#plt.show()


#name = ('1 Agent', '2 Agents', '3 Agents', '4 Agents', '5 Agents', 
#		'6 Agents', '7 Agents', '8 Agents', '9 Agents', '10 Agents')
name1 = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
y_pos = np.arange(len(name1))
performance1 = [0.33, 0.67, 0.74, 0.91, 0.98, 0.98, 0.97, 0.97, 0.97, 0.98 ]

plt.bar(y_pos, performance1, align='center', alpha=0.5, color='g')
plt.xticks(y_pos, name1)

plt.xlabel('Number of agents')
plt.ylabel('Success rate of training')
#plt.yticks(np.arange(0, 1, step=0.1))
plt.yticks(np.linspace(0, 1, num=11))
#plt.title('')

plt.show()

name2 = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
y_pos = np.arange(len(name2))
performance2 = [0.26, 0.69, 0.83, 0.87, 0.97, 0.89, 0.93, 0.92, 0.87, 0.87 ]

plt.bar(y_pos, performance2, align='center', alpha=0.5, color='g')
plt.xticks(y_pos, name2)

plt.xlabel('Number of agents')
plt.ylabel('Success rate of training')
#plt.yticks(np.arange(0, 1, step=0.1))
plt.yticks(np.linspace(0, 1, num=11))
#plt.title('')

plt.show()

