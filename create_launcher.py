#!/usr/bin/env python
# coding: utf-8

import itertools

args = {"-number_agent": [5],
	"-random_seed": list(range(0, 100)),
	"-model_name": ["targetExp225d"]
	}
python_file = "train.py"

keys, values = zip(*args.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(permutations_dicts)

with open("train_array.txt", "w") as f:
	for combination in permutations_dicts:
		line = f"python {python_file} "
		for key in combination:
			line = line + f"{key} {combination[key]} "
		line = line + "\n"
		f.write(line)
