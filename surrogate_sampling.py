import matplotlib.pyplot as plt
import numpy as np
from smt.sampling_methods import BoxBehnken, LHS
import argparse
import json

args = argparse.ArgumentParser()
args.add_argument('--sample_method', type=str, default='lhs')
args.add_argument('--num', type=int, default=5)
args.add_argument('--save_dir', type=str, default='sg_samples')
args = args.parse_args()

sample_method = args.sample_method
num = args.num
save_dir = args.save_dir

xlimits = np.array([[0.0, 1000], [0.0, 1000], [0.0, 1000],[0.0, 1000], [0.0, 1000], [0.0, 1000]])

if sample_method == 'bbk':
    sampling = BoxBehnken(xlimits=xlimits)
elif sample_method == 'lhs':
    sampling = LHS(xlimits=xlimits)
else:
    raise ValueError("Invalid sample method")

x = sampling(num)

samples = []
print(x.shape)

for idx in range(x.shape[0]):
    samples.append({
        'w1': x[idx, 0],
        'w2': x[idx, 1],
        'w3': x[idx, 2],
        'w4': x[idx, 3],
        'w5': x[idx, 4],
        'w6': x[idx, 5],
    })

with open (f'{save_dir}/{sample_method}_{num}.json', 'w') as f:
    json.dump(samples, f)