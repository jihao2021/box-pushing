import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from scipy.stats import sem,tstd 
import argparse
#import math
import constant
from math import sqrt
import collections
from scipy.stats import entropy

parser = argparse.ArgumentParser()
parser.add_argument('-number_agent', type=int, default=1,
                        help='choose number of agents')
parser.add_argument('-seed_range', type=int, default=100, 
            help = 'choose seed range')
parser.add_argument('-type_of_simulation', type=str, default='rotation', 
            help = 'choose type rotation, change_x or change_y')

args = parser.parse_args()
print('=' * 50)
print(args)
print('=' * 50)
Number_agent= args.number_agent
seed_range = args.seed_range
type_of_simulation = args.type_of_simulation
color_list = constant.color_list
random_seed=1
seed_list = np.arange(1,seed_range+1)
data_frame_list_greedy = [[] for _ in range(len(seed_list))]

path = './NUM_AGENT' + str(Number_agent).zfill(2) + '/'

for i, item in enumerate(seed_list):
    data_frame_list_greedy[i]=pd.read_csv(path + 
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


df_final_greedy = data_frame_list_greedy[0]
for i in range(len(data_frame_list_greedy)-1):
  df_final_greedy = pd.concat([df_final_greedy, data_frame_list_greedy[i+1]], ignore_index=True)
df_final_greedy.index = np.arange(1, len(df_final_greedy) + 1)
df_final_numpy_greedy = df_final_greedy.to_numpy()

greedy_list = []
for i in range(len(df_final_numpy_greedy)):
    greedy_list.append(float(df_final_numpy_greedy[i]))

greedy_dictionary = {}
for i in range(len(greedy_list)):
    greedy_dictionary[i+1] = greedy_list[i]
#only use successful seed
sorted_dictionary = sorted(greedy_dictionary.items(), key=lambda kv: kv[1])

ordered_dictionary = collections.OrderedDict(sorted_dictionary)
print(ordered_dictionary)

ordered_dictionary_new = {key:val for key, val in ordered_dictionary.items() if val == 1}
print(ordered_dictionary_new)


list_keys_od = list(ordered_dictionary_new.keys())
list_values_od = list(ordered_dictionary_new.values())


# from 0 to 99(inclusive)
#seed_list_selected = list_keys_od[0:100] 
#seed_list_selected = list_keys_od[0:50]
#seed_list_selected = list_keys_od[50:100]
#seed_list_selected = list_keys_od[0:100]
seed_list_selected = list_keys_od
#print(len(seed_list_selected))


data_frame_list = [[] for _ in range(len(seed_list_selected))]
for i, item in enumerate(seed_list_selected):
    data_frame_list[i]=pd.read_csv(path + 
                                    str(item).zfill(4) + 
                                    '/greedy_output/'+
                                    'check_trajectory'+
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
#print(df_final)
#print(df_final.head())
df_final_numpy = df_final.to_numpy()
print(len(df_final_numpy))
#print(df_final_numpy.head())
#print(df_final_numpy[0])
rotation_list = []
change_x_list=[]
change_y_list=[]
for i in range(len(df_final_numpy)):
    data = list(df_final_numpy[i])
    Number_of_0 = data.count(0)
    Number_of_1 = data.count(1)
    Number_of_2 = data.count(2)
    Number_of_3 = data.count(3)
    Number_of_4 = data.count(4)
    Number_of_5 = data.count(5)
    Sum = Number_of_0 + Number_of_1 + Number_of_2 + Number_of_3 + Number_of_4 + Number_of_5
    rotation = Number_of_0 + Number_of_2 - Number_of_1 - Number_of_3
    change_x = Number_of_2 + Number_of_3 - Number_of_0 - Number_of_1
    change_y = Number_of_4 - Number_of_5
    #print(rotation)
    rotation_list.append(rotation)
    change_x_list.append(change_x)
    change_y_list.append(change_y)



od = collections.OrderedDict()
#for i in range(0, len(sorted_key_list)):
for i in range(-Number_agent, Number_agent + 1):
    if type_of_simulation == 'rotation':
        if i in rotation_list:
            od[i]= rotation_list.count(i)
        else:
            od[i] = 0
    elif type_of_simulation == 'change_x':
        if i in change_x_list:
            od[i]= change_x_list.count(i)
        else:
            od[i] = 0
    elif type_of_simulation == 'change_y':
        if i in change_y_list:
            od[i]= change_y_list.count(i)
        else:
            od[i] = 0
    else:
        print('Error')

print(od)
'''
df_final_numpy = df_final.to_numpy()
distance_mean_list = np.mean(df_final_numpy, axis=0)
distance_sem_list = sem(df_final_numpy).tolist()
print(distance_mean_list)



df_total = pd.DataFrame()
'''

list_keys = list(od.keys())
list_values = list(od.values())
#print(list_keys)
#print(type(list_values))

normed_list_values = [i/sum(list_values) for i in list_values]
print(normed_list_values)
print(entropy(normed_list_values, base=2))

'''
if Number_agent < 11:
    N_new = normed_list_values
    print(N_new)
    print(entropy(N_new, base=2))
else:
    middle = int((len(normed_list_values) - 1)/2)
    print('middle is:', middle)

    #if range is [-5,5]
    #N_new=normed_list_values[middle-5:middle+6]


    #if range is [-4,4]
    #N_new=normed_list_values[middle-4:middle+5]

    #if range is [-3,3]
    #N_new=normed_list_values[middle-3:middle+4]

    #if range is [-2,2]
    N_new=normed_list_values[middle-11:middle+12]


    print(N_new)
    print(entropy(N_new, base=2))
'''
plt.bar(list_keys, normed_list_values, color='g')
plt.show()



