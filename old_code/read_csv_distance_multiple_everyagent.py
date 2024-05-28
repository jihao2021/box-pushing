import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from scipy.stats import sem  
import argparse
import constant

parser = argparse.ArgumentParser()
parser.add_argument('-number_agent', type=int, default=1,
                        help='choose number of agents')
parser.add_argument('-seed_range', type=int, default=100, 
						help = 'choose seed range')
parser.add_argument('-episode_length', type=int, default=8000,
                        help = 'choose episode length')
parser.add_argument('-start_episode_eval', type=int, default=1000,
                        help = 'choose where to start evaluate rank order until last episode')

args = parser.parse_args()
print('=' * 50)
print(args)
print('=' * 50)
Number_agent = args.number_agent
seed_range = args.seed_range
start_episode_eval = args.start_episode_eval
color_list = constant.color_list
episode_length = args.episode_length
episode_list = np.arange(1,episode_length+1)
seed_list = np.arange(1,seed_range+1)
data_frame_list = [[] for _ in range(len(seed_list))]
path = './NUM_AGENT' + str(Number_agent).zfill(2) + '/'
column_list = [[] for _ in range(episode_length)]


for i in range(0,episode_length):
    column_list[i] = str(i)

    
for i, item in enumerate(seed_list):
    data_frame_list[i]=pd.read_csv(path + 
                                    str(item).zfill(4) +
                                    '/distance_output'+
                                    '/total_distance_list'+
                                    '-NUM_AGENT'+
                                    str(Number_agent).zfill(2)+
                                    '-Random_seed'+
                                    str(item).zfill(4)+
                                    '.csv',header=None)
    data_frame_list[i].columns = column_list
    total_rank = data_frame_list[i][str(start_episode_eval)].rank()
    for j in range(start_episode_eval+1,episode_length):
        total_rank += data_frame_list[i][str(j)].rank()
    data_frame_list[i]['Rank'] = total_rank
    data_frame_list[i] = data_frame_list[i].sort_values(['Rank'], ascending=[False])
    #print(data_frame_list[i])



label_list = []
for i in range(Number_agent):
    label_list.append('agent' + str(i+1))

df_final = data_frame_list[0]
for i in range(len(data_frame_list)-1):
	df_final = pd.concat([df_final, data_frame_list[i+1]], ignore_index=True)

#print(df_final.head())
print(df_final)
df_final = df_final.iloc[:,:-1]
print(df_final)
df_final=df_final.to_numpy()

total_distance_list = [[] for _ in range(Number_agent)]
for i in range(Number_agent):
    total_distance_list[i] = np.mean(df_final[i:Number_agent*100:Number_agent],axis=0).tolist()

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

distance_sliding_mean_list = [[] for _ in range(Number_agent)]
for i in range(Number_agent):
    distance_sliding_mean_list[i] = sliding_mean(total_distance_list[i],window=100)


for i in range(Number_agent):
    plt.plot(distance_sliding_mean_list[i], color_list[i],label=label_list[i])
#plt.legend()
#plt.title("box_pushing", fontsize=22)  
plt.legend(loc='upper right')
plt.ylabel('Total distance')
#plt.ylim(top=200)
#plt.xlabel('training episode(every 100 episode)')
plt.xlabel('Training episode')
plt.show()
