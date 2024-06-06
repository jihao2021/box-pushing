import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from scipy.stats import sem,tstd 
import argparse
#import math
import constant
from math import sqrt

parser = argparse.ArgumentParser()
parser.add_argument('-number_agent', type=int, default=1,
                        help='choose number of agents')
parser.add_argument('-seed_range', type=int, default=100, 
            help = 'choose seed range')

args = parser.parse_args()
print('=' * 50)
print(args)
print('=' * 50)
Number_agent= args.number_agent
seed_range = args.seed_range
color_list = constant.color_list
random_seed=1
seed_list = np.arange(1,seed_range+1)
data_frame_list = [[] for _ in range(len(seed_list))]
path = './NUM_AGENT' + str(Number_agent).zfill(2) + '/'
for i, item in enumerate(seed_list):
    data_frame_list[i]=pd.read_csv(path + 
                                    str(item).zfill(4) + 
                                    '/greedy_output/'+
                                    'check_success'+
                                    '-NUM_AGENT'+
                                    str(Number_agent).zfill(2)+
                                    '-Random_seed'+
                                    str(random_seed).zfill(4)+
                                    '-NUMBER_OF_AGENT_TRAINED_MODEL'+
                                    str(Number_agent).zfill(2)+
                                    '-Random_seed_trained_model'+
                                    str(item).zfill(4) +
                                    '.csv',header=None)
label_list = []
for i in range(Number_agent):
    label_list.append('agent' + str(i+1))

df_final = data_frame_list[0]
for i in range(len(data_frame_list)-1):
  df_final = pd.concat([df_final, data_frame_list[i+1]], ignore_index=True)
df_final.index = np.arange(1, len(df_final) + 1)
print(df_final)
print(df_final.head())


df_final_numpy = df_final.to_numpy()
distance_mean_list = np.mean(df_final_numpy, axis=0)
distance_sem_list = sem(df_final_numpy).tolist()
print(distance_mean_list)



df_total = pd.DataFrame()





