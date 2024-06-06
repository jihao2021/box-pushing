import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
###simulation1 folder if you wanna change, do simulatin or other folder in path 

from scipy.stats import sem  
import argparse
import constant

parser = argparse.ArgumentParser()
parser.add_argument('-number_agent', type=int, default=1,
                        help='choose number of agents')
parser.add_argument('-seed_number', type=int, default=100, 
						help = 'choose seed range')
parser.add_argument('-episode_length', type=int, default=8000,
                        help = 'choose episode length')

args = parser.parse_args()
print('=' * 50)
print(args)
print('=' * 50)
Number_agent = args.number_agent
seed_number = args.seed_number
episode_length = args.episode_length

'''
episode_list = np.arange(1,5001)
seed_list = np.arange(1,seed_range)
data_frame_list = [[] for _ in range(len(seed_list))]
for i, item in enumerate(seed_list):
	path = '~/simulation/NUM_AGENT' + str(Number_agent).zfill(2) + '/'
	data_frame_list[i]=pd.read_csv(path + 
									str(item).zfill(4) +
									'/reward_output'+
									'/reward_history'+
									'-NUM_AGENT'+
									str(Number_agent).zfill(2)+
									'-Random_seed'+
									str(item).zfill(4)+
									'.csv',header=None)
'''
episode_list = np.arange(1,episode_length+1)
#Number_agent = 4
color_list = constant.color_list

label_list = []
for i in range(Number_agent):
    label_list.append('agent' + str(i+1))

#seed_list = np.arange(1)
seed_list = list([seed_number])

data_frame_list = [[] for _ in range(len(seed_list))]
path = './NUM_AGENT' + str(Number_agent).zfill(2) + '/'
for i, item in enumerate(seed_list):
    #print(i)
    data_frame_list[0]=pd.read_csv(path + 
                                    str(item).zfill(4) +
                                    '/distance_output'+
                                    '/total_distance_list'+
                                    '-NUM_AGENT'+
                                    str(Number_agent).zfill(2)+
                                    '-Random_seed'+
                                    str(item).zfill(4)+
                                    '.csv',header=None)

df_final = data_frame_list[0]
for i in range(len(data_frame_list)-1):
	df_final = pd.concat([df_final, data_frame_list[i+1]], ignore_index=True)

print(df_final)
print(len(df_final.iloc[0].tolist()))

total_distance_list = [[] for _ in range(Number_agent)]
for i in range(Number_agent):
    total_distance_list[i] = df_final.iloc[i].tolist()

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

#reward_sem_list_sliding_mean = reward_sem_list
#reward_mean_list_sliding_mean = reward_mean_list

distance_sliding_mean_list = [[] for _ in range(Number_agent)]
for i in range(Number_agent):
    distance_sliding_mean_list[i] = sliding_mean(total_distance_list[i],window=100)



#reward_sem_list_sliding_mean = sliding_mean(reward_sem_list, window=100)
#reward_mean_list_sliding_mean = sliding_mean(reward_mean_list, window=100)

#plt.fill_between(episode_list, reward_mean_list_sliding_mean - reward_sem_list_sliding_mean,  
#                reward_mean_list_sliding_mean + reward_sem_list_sliding_mean, color="yellow")  
#plt.plot(episode_list, reward_mean_list_sliding_mean, color="magenta", lw=1)  
for i in range(Number_agent):
    plt.plot(distance_sliding_mean_list[i], color_list[i],label=label_list[i])
#plt.legend()
#plt.title("box_pushing", fontsize=22)  
plt.legend(loc='upper right')
plt.ylabel('Total Step Sizes')
#plt.ylim(top=200)
#plt.xlabel('training episode(every 100 episode)')
plt.xlabel('training episode')
plt.show()