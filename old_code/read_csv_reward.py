import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from scipy.stats import sem 
import constant 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-number_agent', type=int, default=1,
                        help='choose number of agents')
parser.add_argument('-seed_range', type=int, default=100, 
						help = 'choose seed range')
parser.add_argument('-episode_length', type=int, default=8000,
                        help = 'choose episode length')

args = parser.parse_args()
print('=' * 50)
print(args)
print('=' * 50)
Number_agent = args.number_agent
seed_range = args.seed_range
episode_length = args.episode_length
episode_list = np.arange(1,episode_length+1)
seed_list = np.arange(1,seed_range+1)
data_frame_list = [[] for _ in range(len(seed_list))]

color_list = constant.color_list

label_list = []
for i in range(Number_agent):
    label_list.append('agent' + str(i+1))

for i, item in enumerate(seed_list):
	path = './NUM_AGENT' + str(Number_agent).zfill(2) + '/'
	data_frame_list[i]=pd.read_csv(path + 
									str(item).zfill(4) +
									'/reward_output'+
									'/reward_history'+
									'-NUM_AGENT'+
									str(Number_agent).zfill(2)+
									'-Random_seed'+
									str(item).zfill(4)+
									'.csv',header=None)

df_final = data_frame_list[0]
for i in range(len(data_frame_list)-1):
	df_final = pd.concat([df_final, data_frame_list[i+1]], ignore_index=True)

print(df_final.head())

df_final_numpy = df_final.to_numpy()
reward_mean_list = np.mean(df_final_numpy, axis=0)
reward_sem_list = sem(df_final_numpy).tolist()

###Reward plot based on average 100 episodes, this turns out to be same/similar to sliding mean, so no need to plot this
#rMat = np.resize(np.array(reward_mean_list),[len(reward_mean_list) // 100, 100] )
#rmean = np.average(rMat, 1)
#plt.plot(rmean)
#plt.show()

def sliding_mean(data_array, window=5):  
    data_array = np.array(data_array)  
    new_list = []  
    for i in range(len(data_array)):  
        indices = range(max(i - window + 1, 0),  
                        min(i + window + 1, len(data_array)))  
        avg = 0  
        for j in indices:  
            avg += data_array[j]  
        avg /= float(len(indices))  
        new_list.append(avg)        
    return np.array(new_list)

reward_sem_list_sliding_mean = sliding_mean(reward_sem_list, window=100)
reward_mean_list_sliding_mean = sliding_mean(reward_mean_list, window=100)

reward_sliding_mean_listagent = [[] for _ in range(Number_agent)]
for i in range(Number_agent):
    reward_sliding_mean_listagent[i] = reward_mean_list_sliding_mean

plt.fill_between(episode_list, reward_mean_list_sliding_mean - reward_sem_list_sliding_mean,  
                reward_mean_list_sliding_mean + reward_sem_list_sliding_mean, color="palegreen")  
#plt.plot(episode_list, reward_mean_list_sliding_mean, color="magenta", lw=1)  
for i in range(Number_agent):
    plt.plot(episode_list, reward_sliding_mean_listagent[i], color_list[i],label=label_list[i])

plt.legend(loc='lower right')
#plt.title("box_pushing", fontsize=22)  
plt.ylabel('Reward')
plt.ylim(top=700)
#plt.xlabel('training episode(every 100 episode)')
plt.xlabel('Training episode')
plt.show()

#rMat_sem = np.resize(np.array(reward_sem_list),[len(reward_sem_list) // 100, 100] )
#rmean_sem = np.average(rMat_sem, 1)
#plt.plot(rmean_sem)
#plt.ylim()

#plt.show()

plt.plot(episode_list, reward_sem_list_sliding_mean, color='magenta')
#plt.ylim(top=40)
plt.show()
