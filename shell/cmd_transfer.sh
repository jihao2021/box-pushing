#!/bin/bash
cd "$pwd"

num_agent='1'
echo "NUM_AGNET is $num_agent"

random_seed='1'

for number_agent_trained_model in $(seq 1 10)
do
	echo "$number_agent_trained_model"
	for seed in $(seq 1 10)
	do
		echo "$seed"
		python3 run_this_semigreedy.py -number_agent "$num_agent" -random_seed "$random_seed" -number_agent_trained_model "$number_agent_trained_model" -random_seed_trained_model "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $random_seed)/out_transfer_NUMBER_AGENT_TRAINED_MODEL$(printf %02d $number_agent_trained_model)random_seed_trained_model$(printf %04d $seed)" &

	done
	wait
done

num_agent='2'
echo "NUM_AGNET is $num_agent"

random_seed='1'

for number_agent_trained_model in $(seq 1 10)
do
	echo "$number_agent_trained_model"
	for seed in $(seq 1 10)
	do
		echo "$seed"
		python3 run_this_semigreedy.py -number_agent "$num_agent" -random_seed "$random_seed" -number_agent_trained_model "$number_agent_trained_model" -random_seed_trained_model "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $random_seed)/out_transfer_NUMBER_AGENT_TRAINED_MODEL$(printf %02d $number_agent_trained_model)random_seed_trained_model$(printf %04d $seed)" &

	done
	wait
done

num_agent='3'
echo "NUM_AGNET is $num_agent"

random_seed='1'

for number_agent_trained_model in $(seq 1 10)
do
	echo "$number_agent_trained_model"
	for seed in $(seq 1 10)
	do
		echo "$seed"
		python3 run_this_semigreedy.py -number_agent "$num_agent" -random_seed "$random_seed" -number_agent_trained_model "$number_agent_trained_model" -random_seed_trained_model "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $random_seed)/out_transfer_NUMBER_AGENT_TRAINED_MODEL$(printf %02d $number_agent_trained_model)random_seed_trained_model$(printf %04d $seed)" &

	done
	wait
done



num_agent='4'
echo "NUM_AGNET is $num_agent"
random_seed='1'
for number_agent_trained_model in $(seq 1 10)
do
	echo "$number_agent_trained_model"
	for seed in $(seq 1 10)
	do
		echo "$seed"
		python3 run_this_semigreedy.py -number_agent "$num_agent" -random_seed "$random_seed" -number_agent_trained_model "$number_agent_trained_model" -random_seed_trained_model "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $random_seed)/out_transfer_NUMBER_AGENT_TRAINED_MODEL$(printf %02d $number_agent_trained_model)random_seed_trained_model$(printf %04d $seed)" &

	done
	wait
done

num_agent='5'
echo "NUM_AGNET is $num_agent"

random_seed='1'

for number_agent_trained_model in $(seq 1 10)
do
	echo "$number_agent_trained_model"
	for seed in $(seq 1 10)
	do
		echo "$seed"
		python3 run_this_semigreedy.py -number_agent "$num_agent" -random_seed "$random_seed" -number_agent_trained_model "$number_agent_trained_model" -random_seed_trained_model "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $random_seed)/out_transfer_NUMBER_AGENT_TRAINED_MODEL$(printf %02d $number_agent_trained_model)random_seed_trained_model$(printf %04d $seed)" &

	done
	wait
done

num_agent='6'
echo "NUM_AGNET is $num_agent"

random_seed='1'

for number_agent_trained_model in $(seq 1 10)
do
	echo "$number_agent_trained_model"
	for seed in $(seq 1 10)
	do
		echo "$seed"
		python3 run_this_semigreedy.py -number_agent "$num_agent" -random_seed "$random_seed" -number_agent_trained_model "$number_agent_trained_model" -random_seed_trained_model "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $random_seed)/out_transfer_NUMBER_AGENT_TRAINED_MODEL$(printf %02d $number_agent_trained_model)random_seed_trained_model$(printf %04d $seed)" &

	done
	wait
done


num_agent='7'
echo "NUM_AGNET is $num_agent"

random_seed='1'

for number_agent_trained_model in $(seq 1 10)
do
	echo "$number_agent_trained_model"
	for seed in $(seq 1 10)
	do
		echo "$seed"
		python3 run_this_semigreedy.py -number_agent "$num_agent" -random_seed "$random_seed" -number_agent_trained_model "$number_agent_trained_model" -random_seed_trained_model "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $random_seed)/out_transfer_NUMBER_AGENT_TRAINED_MODEL$(printf %02d $number_agent_trained_model)random_seed_trained_model$(printf %04d $seed)" &

	done
	wait
done


num_agent='8'
echo "NUM_AGNET is $num_agent"

random_seed='1'

for number_agent_trained_model in $(seq 1 10)
do
	echo "$number_agent_trained_model"
	for seed in $(seq 1 10)
	do
		echo "$seed"
		python3 run_this_semigreedy.py -number_agent "$num_agent" -random_seed "$random_seed" -number_agent_trained_model "$number_agent_trained_model" -random_seed_trained_model "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $random_seed)/out_transfer_NUMBER_AGENT_TRAINED_MODEL$(printf %02d $number_agent_trained_model)random_seed_trained_model$(printf %04d $seed)" &

	done
	wait
done


num_agent='9'
echo "NUM_AGNET is $num_agent"

random_seed='1'

for number_agent_trained_model in $(seq 1 10)
do
	echo "$number_agent_trained_model"
	for seed in $(seq 1 10)
	do
		echo "$seed"
		python3 run_this_semigreedy.py -number_agent "$num_agent" -random_seed "$random_seed" -number_agent_trained_model "$number_agent_trained_model" -random_seed_trained_model "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $random_seed)/out_transfer_NUMBER_AGENT_TRAINED_MODEL$(printf %02d $number_agent_trained_model)random_seed_trained_model$(printf %04d $seed)" &

	done
	wait
done


num_agent='10'
echo "NUM_AGNET is $num_agent"

random_seed='1'

for number_agent_trained_model in $(seq 1 10)
do
	echo "$number_agent_trained_model"
	for seed in $(seq 1 10)
	do
		echo "$seed"
		python3 run_this_semigreedy.py -number_agent "$num_agent" -random_seed "$random_seed" -number_agent_trained_model "$number_agent_trained_model" -random_seed_trained_model "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $random_seed)/out_transfer_NUMBER_AGENT_TRAINED_MODEL$(printf %02d $number_agent_trained_model)random_seed_trained_model$(printf %04d $seed)" &

	done
	wait
done






