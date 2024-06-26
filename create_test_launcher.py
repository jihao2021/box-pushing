#!/usr/bin/env python
# coding: utf-8

import itertools

args = {"-number_agent": [5],
	"-random_seed": [0],
	"-number_agent_trained_model": [5],
	"-random_seed_trained_model": list(range(0, 100)),
	"-model_name": ["targetExp225d"]}
python_file = "run_this_greedy.py"

keys, values = zip(*args.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(permutations_dicts)

with open("test_array.txt", "w") as f:
	for combination in permutations_dicts:
		line = f"python {python_file} "
		for key in combination:
			line = line + f"{key} {combination[key]} "
		line = line + "\n"
		f.write(line)
