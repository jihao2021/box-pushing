#!/bin/bash
cd "$pwd"
random_seed='1'

transfer_job_name='transfer_job_oneagent'
rm -f "$transfer_job_name"
touch "$transfer_job_name"


for num_agent in $(seq 1 10)
do
	echo "NUM_AGNET is $num_agent"
	for number_agent_trained_model in $(seq 1 1)
	do
		echo "$number_agent_trained_model"
		for seed in $(seq 101 200)
		do
			echo "$seed"
			echo "python3 run_this_semigreedy.py -number_agent "$num_agent" -random_seed "$random_seed" -number_agent_trained_model "$number_agent_trained_model" -random_seed_trained_model "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $random_seed)/out_transfer_NUMBER_AGENT_TRAINED_MODEL$(printf %02d $number_agent_trained_model)random_seed_trained_model$(printf %04d $seed)"" >> "$transfer_job_name"

		done
	done
done





