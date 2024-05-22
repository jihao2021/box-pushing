#!/bin/bash
cd "$pwd"
random_seed='1'

job_name='greedy_job'
rm -f "$job_name"
touch "$job_name"

for num_agent in $(seq 1 12)
do
	for seed in $(seq 1 10)
	do
		echo "$seed"
		echo "python3 run_this_greedy.py -number_agent "$num_agent" -random_seed "$random_seed"  -number_agent_trained_model "$num_agent" -random_seed_trained_model "$seed"" >> "$job_name"

	done


	for seed in $(seq 11 20)
	do
		echo "$seed"
		echo "python3 run_this_greedy.py -number_agent "$num_agent" -random_seed "$random_seed"  -number_agent_trained_model "$num_agent" -random_seed_trained_model "$seed"" >> "$job_name"

	done


	for seed in $(seq 21 30)
	do
		echo "$seed"
		echo "python3 run_this_greedy.py -number_agent "$num_agent" -random_seed "$random_seed"  -number_agent_trained_model "$num_agent" -random_seed_trained_model "$seed"" >> "$job_name"

	done


	for seed in $(seq 31 40)
	do
		echo "$seed"
		echo "python3 run_this_greedy.py -number_agent "$num_agent" -random_seed "$random_seed"  -number_agent_trained_model "$num_agent" -random_seed_trained_model "$seed"" >> "$job_name"

	done


	for seed in $(seq 41 50)
	do
		echo "$seed"
		echo "python3 run_this_greedy.py -number_agent "$num_agent" -random_seed "$random_seed"  -number_agent_trained_model "$num_agent" -random_seed_trained_model "$seed"" >> "$job_name"

	done

	for seed in $(seq 51 60)
	do
		echo "$seed"
		echo "python3 run_this_greedy.py -number_agent "$num_agent" -random_seed "$random_seed"  -number_agent_trained_model "$num_agent" -random_seed_trained_model "$seed"" >> "$job_name"

	done


	for seed in $(seq 61 70)
	do
		echo "$seed"
		echo "python3 run_this_greedy.py -number_agent "$num_agent" -random_seed "$random_seed"  -number_agent_trained_model "$num_agent" -random_seed_trained_model "$seed"" >> "$job_name"

	done

	for seed in $(seq 71 80)
	do
		echo "$seed"
		echo "python3 run_this_greedy.py -number_agent "$num_agent" -random_seed "$random_seed"  -number_agent_trained_model "$num_agent" -random_seed_trained_model "$seed"" >> "$job_name"
	done

	for seed in $(seq 81 90)
	do
		echo "$seed"
		echo "python3 run_this_greedy.py -number_agent "$num_agent" -random_seed "$random_seed"  -number_agent_trained_model "$num_agent" -random_seed_trained_model "$seed"" >> "$job_name"
	done

	for seed in $(seq 91 100)
	do
		echo "$seed"
		echo "python3 run_this_greedy.py -number_agent "$num_agent" -random_seed "$random_seed"  -number_agent_trained_model "$num_agent" -random_seed_trained_model "$seed"" >> "$job_name"
	done
done
