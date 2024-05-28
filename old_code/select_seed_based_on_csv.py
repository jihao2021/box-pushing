import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from scipy.stats import sem  
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-number_agent', type=int, default=1,
                        help='choose number of agents')
parser.add_argument('-random_seed', type=int, default=1,
						help='choose random seed')
parser.add_argument('-seed_range', type=int, default=100, 
						help = 'choose seed range')
parser.add_argument('-number_of_calculation', type=int, default=50,
                        help='choose number of ranked seed as calculation')
parser.add_argument('-number_of_agent_trained_model',type=int, default=1,
						help = 'choose number of agent trained model')
parser.add_argument('-episode_length', type=int, default=8000,
                        help = 'choose episode length')



args = parser.parse_args()
print('=' * 50)
print(args)
print('=' * 50)
Number_agent = args.number_agent
seed_range = args.seed_range
episode_length = args.episode_length
random_seed = args.random_seed
NUMBER_OF_AGENT_TRAINED_MODEL = args.number_of_agent_trained_model
number_of_calculation = args.number_of_calculation
episode_list = np.arange(1,episode_length+1)
seed_list = np.arange(1,seed_range+1)
data_frame_list = [[] for _ in range(len(seed_list))]

for i, item in enumerate(seed_list):
	path = './NUM_AGENT' + str(Number_agent).zfill(2) + '/' + str(random_seed).zfill(4) + '/semigreedy_output/'+ str(NUMBER_OF_AGENT_TRAINED_MODEL).zfill(2)+ '/'
	data_frame_list[i]=pd.read_csv(path + 
									'reach_goal_history'+
                           			'-NUM_AGENT'+
                           			str(Number_agent).zfill(2)+
                           			'-Random_seed'+
                           			str(random_seed).zfill(4)+
                           			'-NUMBER_OF_AGENT_TRAINED_MODEL'+
                           			str(NUMBER_OF_AGENT_TRAINED_MODEL).zfill(2)+
                           			'-Random_seed_trained_model'+
                           			str(item).zfill(4)+
                           			'.csv',header=None)
#print(data_frame_list)
df_final = data_frame_list[0]
for i in range(len(data_frame_list)-1):
	df_final = pd.concat([df_final, data_frame_list[i+1]], ignore_index=True)

#print(df_final)
df_final.columns=['success_rate1', 'success_rate2', 'success_rate3']

print(df_final)

df_final.index = np.arange(1, len(df_final) + 1)

df_final = df_final.sort_values(['success_rate1', 'success_rate2', 'success_rate3'], ascending=[False, False, False])
print(df_final)
success_rate1_list = np.array(df_final.head(number_of_calculation)['success_rate1'].tolist())
success_rate2_list = np.array(df_final.head(number_of_calculation)['success_rate2'].tolist())
success_rate3_list = np.array(df_final.head(number_of_calculation)['success_rate3'].tolist())
#print(success_rate1_list)
std_successrate1 = round(round(np.std(success_rate1_list),4)*100,2)
std_successrate2 = round(round(np.std(success_rate2_list),4)*100,2)
std_successrate3 = round(round(np.std(success_rate3_list),4)*100,2)
mean_successrate1 = round(round(np.mean(success_rate1_list),4)*100,2)
mean_successrate2 = round(round(np.mean(success_rate2_list),4)*100,2)
mean_successrate3 = round(round(np.mean(success_rate3_list),4)*100,2)

print(mean_successrate1, mean_successrate2, mean_successrate3)
print(std_successrate1, std_successrate2, std_successrate3)

      #print(std_successrate1)
'''
df_final['Number_agent'] = str(Number_agent).zfill(2)
df_final['Random_seed'] = str(random_seed).zfill(4)
df_final['Number_of_agent_trained_model'] = str(NUMBER_OF_AGENT_TRAINED_MODEL).zfill(2)
df_final['std_successrate1'] = std_successrate1
df_final['std_successrate2'] = std_successrate2
df_final['std_successrate3'] = std_successrate3
df_final['mean_successrate1'] = mean_successrate1
df_final['mean_successrate2'] = mean_successrate2
df_final['mean_successrate3'] = mean_successrate3

#df_final = df_final.sort_values(['success_rate1', 'success_rate2'], ascending=[False, False])
#print(df_final.head(10))

df_total_simplified = df_final[['Number_agent', 'Random_seed', 'Number_of_agent_trained_model', 
                                          'std_successrate1', 'std_successrate2', 'std_successrate3',
                                          'mean_successrate1', 'mean_successrate2', 'mean_successrate3']]

df_total_show = df_total_simplified[['Number_agent', 'Random_seed', 'Number_of_agent_trained_model',
                                     'std_successrate1', 'mean_successrate1',
                                     'std_successrate2', 'mean_successrate2',
                                     'std_successrate3','mean_successrate3']]

df_total_show = df_total_show.applymap(str)
df_total_show['success_rate1_total'] = df_total_show['mean_successrate1'] + '%' + ("\u00B1") + df_total_show['std_successrate1'] + '%'
df_total_show['success_rate2_total'] = df_total_show['mean_successrate2'] + '%' + ("\u00B1") + df_total_show['std_successrate2'] + '%'
df_total_show['success_rate3_total'] = df_total_show['mean_successrate3'] + '%' + ("\u00B1") + df_total_show['std_successrate3'] + '%'
'''

