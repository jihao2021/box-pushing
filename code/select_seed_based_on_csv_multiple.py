import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from scipy.stats import sem,tstd 
import argparse
#import math
from math import sqrt


#from scipy import stats
'''
parser = argparse.ArgumentParser()
parser.add_argument('-number_agent', type=int, default=1,
                        help='choose number of agents')
parser.add_argument('-random_seed', type=int, default=1,
						help='choose random seed')
parser.add_argument('-seed_range', type=int, default=100, 
						help = 'choose seed range')
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
episode_list = np.arange(1,episode_length+1)
seed_list = np.arange(1,seed_range+1)
'''
#number_of_calculation= 100
parser = argparse.ArgumentParser()
parser.add_argument('-number_of_calculation', type=int, default=50,
                        help='choose number of ranked seed as calculation')
parser.add_argument('-number_of_total_agent', type=int, default=10,
                        help = 'choose total number of agent in range 10 means 1-10, 12 means 1-12')
args = parser.parse_args()
print('=' * 50)
print(args)
print('=' * 50)
number_of_calculation = args.number_of_calculation
number_of_total_agent = args.number_of_total_agent
df_total = pd.DataFrame()

for i in np.arange (1, number_of_total_agent+1):
   #print(i)
   Number_agent = i
   seed_range= 100
   episode_length = 8000
   random_seed=1
   episode_list=np.arange(1,episode_length+1)
   seed_list=np.arange(1, seed_range+1)
   #df_total = pd.DataFrame(columns=['success_rate1', 'success_rate2', 'success_rate3'])
   for j in np.arange(1,number_of_total_agent + 1):
      NUMBER_OF_AGENT_TRAINED_MODEL=j
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
      df_final = data_frame_list[0]
      for i in range(len(data_frame_list)-1):
      	df_final = pd.concat([df_final, data_frame_list[i+1]], ignore_index=True)

      df_final.columns=['success_rate1', 'success_rate2', 'success_rate3']
      df_final.index = np.arange(1, len(df_final) + 1)

      df_final = df_final.sort_values(['success_rate1', 'success_rate2', 'success_rate3'], ascending=[False, False, False])
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
      #print(std_successrate1)
      df_final['Number_agent'] = str(Number_agent).zfill(2)
      df_final['Random_seed'] = str(random_seed).zfill(4)
      df_final['Number_of_agent_trained_model'] = str(NUMBER_OF_AGENT_TRAINED_MODEL).zfill(2)
      df_final['std_successrate1'] = std_successrate1
      df_final['std_successrate2'] = std_successrate2
      df_final['std_successrate3'] = std_successrate3
      df_final['mean_successrate1'] = mean_successrate1
      df_final['mean_successrate2'] = mean_successrate2
      df_final['mean_successrate3'] = mean_successrate3
      df_total = pd.concat([df_total, df_final.head(number_of_calculation)])
      #print(df_total)
#print(df_total)
#print(df_total.head(100))
#df_total.to_csv ('./select_seed_based_on_csv_multiple.csv', header=True) 

df_total_simplified = df_total[0: (number_of_total_agent **2) * number_of_calculation:number_of_calculation]
df_total_simplified = df_total_simplified[['Number_agent', 'Random_seed', 'Number_of_agent_trained_model', 
                                          'std_successrate1', 'std_successrate2', 'std_successrate3',
                                          'mean_successrate1', 'mean_successrate2', 'mean_successrate3']]
#print(df_total_simplified)

#df_total_show= pd.DataFrame()

#data_frame_list = [[] for _ in range(len(seed_list))]
df_total_show = df_total_simplified[['Number_agent', 'Random_seed', 'Number_of_agent_trained_model',
                                     'std_successrate1', 'mean_successrate1',
                                     'std_successrate2', 'mean_successrate2',
                                     'std_successrate3','mean_successrate3']]
#da_total_show = 
df_total_show = df_total_show.applymap(str)
df_total_show['success_rate1_total'] = df_total_show['mean_successrate1'] + '%' + ("\u00B1") + df_total_show['std_successrate1'] + '%'
df_total_show['success_rate2_total'] = df_total_show['mean_successrate2'] + '%' + ("\u00B1") + df_total_show['std_successrate2'] + '%'
df_total_show['success_rate3_total'] = df_total_show['mean_successrate3'] + '%' + ("\u00B1") + df_total_show['std_successrate3'] + '%'


#print(df_total_show)

final_list1= df_total_show['success_rate1_total'].tolist()
final_list1 = np.array(final_list1)
#print(final_list1.shape)
final_list1 = final_list1.reshape(number_of_total_agent, number_of_total_agent)
final_list1 = final_list1.transpose()
#print(final_list1)

final_list2 = df_total_show['success_rate2_total'].tolist()
final_list2 = np.array(final_list2)
#print(final_list2.shape)
final_list2 = final_list2.reshape(number_of_total_agent, number_of_total_agent)
final_list2 = final_list2.transpose()
#print(final_list2)

final_list3 = df_total_show['success_rate3_total'].tolist()
final_list3 = np.array(final_list3)
#print(final_list3.shape)
final_list3 = final_list3.reshape(number_of_total_agent, number_of_total_agent)
final_list3 = final_list3.transpose()
#print(final_list3)


if number_of_total_agent == 9: 
   dataset1 = pd.DataFrame({'Column1': final_list1[:, 0], 
                           'Column2': final_list1[:, 1],
                           'Column3': final_list1[:, 2],
                           'Column4': final_list1[:, 3],
                           'Column5': final_list1[:, 4],
                           'Column6': final_list1[:, 5],
                           'Column7': final_list1[:, 6],
                           'Column8': final_list1[:, 7],
                           'Column9': final_list1[:, 8],
                           
                                 })
   #print(dataset1)

   dataset2 = pd.DataFrame({'Column1': final_list2[:, 0], 
                           'Column2': final_list2[:, 1],
                           'Column3': final_list2[:, 2],
                           'Column4': final_list2[:, 3],
                           'Column5': final_list2[:, 4],
                           'Column6': final_list2[:, 5],
                           'Column7': final_list2[:, 6],
                           'Column8': final_list2[:, 7],
                           'Column9': final_list2[:, 8],
                           
                                 })
   #print(dataset2)


   dataset3 = pd.DataFrame({'Column1': final_list3[:, 0], 
                           'Column2': final_list3[:, 1],
                           'Column3': final_list3[:, 2],
                           'Column4': final_list3[:, 3],
                           'Column5': final_list3[:, 4],
                           'Column6': final_list3[:, 5],
                           'Column7': final_list3[:, 6],
                           'Column8': final_list3[:, 7],
                           'Column9': final_list3[:, 8],
                           
                                 })


if number_of_total_agent == 10: 
   dataset1 = pd.DataFrame({'Column1': final_list1[:, 0], 
                           'Column2': final_list1[:, 1],
                           'Column3': final_list1[:, 2],
                           'Column4': final_list1[:, 3],
                           'Column5': final_list1[:, 4],
                           'Column6': final_list1[:, 5],
                           'Column7': final_list1[:, 6],
                           'Column8': final_list1[:, 7],
                           'Column9': final_list1[:, 8],
                           'Column10': final_list1[:, 9],
                                 })
   #print(dataset1)

   dataset2 = pd.DataFrame({'Column1': final_list2[:, 0], 
                           'Column2': final_list2[:, 1],
                           'Column3': final_list2[:, 2],
                           'Column4': final_list2[:, 3],
                           'Column5': final_list2[:, 4],
                           'Column6': final_list2[:, 5],
                           'Column7': final_list2[:, 6],
                           'Column8': final_list2[:, 7],
                           'Column9': final_list2[:, 8],
                           'Column10': final_list2[:, 9],
                                 })
   #print(dataset2)


   dataset3 = pd.DataFrame({'Column1': final_list3[:, 0], 
                           'Column2': final_list3[:, 1],
                           'Column3': final_list3[:, 2],
                           'Column4': final_list3[:, 3],
                           'Column5': final_list3[:, 4],
                           'Column6': final_list3[:, 5],
                           'Column7': final_list3[:, 6],
                           'Column8': final_list3[:, 7],
                           'Column9': final_list3[:, 8],
                           'Column10': final_list3[:, 9],
                                 })
   #print(dataset3)

if number_of_total_agent == 12: 
   dataset1 = pd.DataFrame({'Column1': final_list1[:, 0], 
                           'Column2': final_list1[:, 1],
                           'Column3': final_list1[:, 2],
                           'Column4': final_list1[:, 3],
                           'Column5': final_list1[:, 4],
                           'Column6': final_list1[:, 5],
                           'Column7': final_list1[:, 6],
                           'Column8': final_list1[:, 7],
                           'Column9': final_list1[:, 8],
                           'Column10': final_list1[:, 9],
                           'Column11':final_list1[:,10],
                           'Column12': final_list1[:,11],
                                 })
   #print(dataset1)

   dataset2 = pd.DataFrame({'Column1': final_list2[:, 0], 
                           'Column2': final_list2[:, 1],
                           'Column3': final_list2[:, 2],
                           'Column4': final_list2[:, 3],
                           'Column5': final_list2[:, 4],
                           'Column6': final_list2[:, 5],
                           'Column7': final_list2[:, 6],
                           'Column8': final_list2[:, 7],
                           'Column9': final_list2[:, 8],
                           'Column10': final_list2[:, 9],
                           'Column11':final_list2[:,10],
                           'Column12': final_list2[:,11],
                                 })
   #print(dataset2)


   dataset3 = pd.DataFrame({'Column1': final_list3[:, 0], 
                           'Column2': final_list3[:, 1],
                           'Column3': final_list3[:, 2],
                           'Column4': final_list3[:, 3],
                           'Column5': final_list3[:, 4],
                           'Column6': final_list3[:, 5],
                           'Column7': final_list3[:, 6],
                           'Column8': final_list3[:, 7],
                           'Column9': final_list3[:, 8],
                           'Column10': final_list3[:, 9],
                           'Column11':final_list3[:,10],
                           'Column12': final_list3[:,11],
                                 })
   #print(dataset3) 

import os
path = './transfer_results'
if not os.path.exists(path):
   os.makedirs(path)


dataset1.index = np.arange(1, len(dataset1) + 1)
dataset1.to_csv (path + '/select_seed_based_on_csv_multiple_dataset1-' + str(number_of_calculation) + '.csv', header=True) 


dataset2.index = np.arange(1, len(dataset2) + 1)
dataset2.to_csv (path + '/select_seed_based_on_csv_multiple_dataset2-' + str(number_of_calculation) + '.csv', header=True) 


dataset3.index = np.arange(1, len(dataset3) + 1)
dataset3.to_csv (path + '/select_seed_based_on_csv_multiple_dataset3-' + str(number_of_calculation) + '.csv', header=True) 


print('Finished')




