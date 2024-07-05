import argparse
import json

args = argparse.ArgumentParser()
args.add_argument('--number_agent', type=int, default=5)
args.add_argument('--number_agent_trained_model', type=int, default=5)
args.add_argument('--random_seed_trained_model', type=int, default=5)
args.add_argument('--num_episodes', type=int, default=5)
args.add_argument('--params_id', type=str, default='lhs_5')
args.add_argument("--batch_id", type=int, default=1)
args.add_argument("--batch_size", type=int, default=50)

args = args.parse_args()
params_id = args.params_id
n_agents = args.number_agent
n_agents_trained_model = args.number_agent_trained_model
num_episodes = args.num_episodes
random_seed_trained_model = args.random_seed_trained_model
batch_id = args.batch_id
batch_size = args.batch_size


with open(f'./sg_samples/{params_id}.json', 'r') as f:
    samples = json.load(f)

with open(f'sg_scripts/sg_{params_id}_{batch_id}_{batch_size}.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    for idx in range((batch_id - 1)*batch_size, batch_id*batch_size):
        params = samples[idx]
        w1 = params['w1']
        w2 = params['w2']
        w3 = params['w3']
        w4 = params['w4']
        w5 = params['w5']
        w6 = params['w6']
        for random_seed in range(10):
            f.write(f'python train.py -number_agent {n_agents} -random_seed {random_seed} '
                    f'-number_agent_trained_model {n_agents_trained_model} -random_seed_trained_model {random_seed_trained_model} '
                    f'-num_episodes {num_episodes} -rotation_reward True -w1 {w1} -w2 {w2} -w3 {w3} -w4 {w4} -w5 {w5} -w6 {w6} '
                    f'-param_id "{params_id}\\{idx}"\n'
            )

