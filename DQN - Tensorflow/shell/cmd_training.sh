cd ~/simulation1

num_agent='1'
echo "NUM_AGNET is $num_agent"

for num_agent in $(seq 1 10)
do
	for seed in $(seq 1 10)
	do
		echo "$seed"
		python3 run_this.py -number_agent "$num_agent" -random_seed "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $seed)/out" &
		#python3 run_this.py -number_agent 4 -random_seed "$seed" > "./NUM_AGENT04/0001/out" &

	done
	wait

	for seed in $(seq 11 20)
	do
		echo "$seed"
		#num_agent='4'
		python3 run_this.py -number_agent "$num_agent" -random_seed "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $seed)/out" &
		#python3 run_this.py -number_agent 4 -random_seed "$seed" > "./NUM_AGENT04/0001/out" &

	done
	wait

	for seed in $(seq 21 30)
	do
		echo "$seed"
		#num_agent='4'
		python3 run_this.py -number_agent "$num_agent" -random_seed "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $seed)/out" &
		#python3 run_this.py -number_agent 4 -random_seed "$seed" > "./NUM_AGENT04/0001/out" &

	done
	wait

	for seed in $(seq 31 40)
	do
		echo "$seed"
		#num_agent='4'
		python3 run_this.py -number_agent "$num_agent" -random_seed "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $seed)/out" &
		#python3 run_this.py -number_agent 4 -random_seed "$seed" > "./NUM_AGENT04/0001/out" &

	done
	wait

	for seed in $(seq 41 50)
	do
		echo "$seed"
		#num_agent='4'
		python3 run_this.py -number_agent "$num_agent" -random_seed "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $seed)/out" &
		#python3 run_this.py -number_agent 4 -random_seed "$seed" > "./NUM_AGENT04/0001/out" &

	done
	wait

	for seed in $(seq 51 60)
	do
		echo "$seed"
		#num_agent='4'
		python3 run_this.py -number_agent "$num_agent" -random_seed "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $seed)/out" &
		#python3 run_this.py -number_agent 4 -random_seed "$seed" > "./NUM_AGENT04/0001/out" &

	done
	wait

	for seed in $(seq 61 70)
	do
		echo "$seed"
		#num_agent='4'
		python3 run_this.py -number_agent "$num_agent" -random_seed "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $seed)/out" &
		#python3 run_this.py -number_agent 4 -random_seed "$seed" > "./NUM_AGENT04/0001/out" &

	done
	wait

	for seed in $(seq 71 80)
	do
		echo "$seed"
		#num_agent='4'
		python3 run_this.py -number_agent "$num_agent" -random_seed "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $seed)/out" &
		#python3 run_this.py -number_agent 4 -random_seed "$seed" > "./NUM_AGENT04/0001/out" &

	done
	wait

	for seed in $(seq 81 90)
	do
		echo "$seed"
		#num_agent='4'
		python3 run_this.py -number_agent "$num_agent" -random_seed "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $seed)/out" &
		#python3 run_this.py -number_agent 4 -random_seed "$seed" > "./NUM_AGENT04/0001/out" &

	done
	wait

	for seed in $(seq 91 100)
	do
		echo "$seed"
		#num_agent='4'
		python3 run_this.py -number_agent "$num_agent" -random_seed "$seed" > "./NUM_AGENT$(printf %02d $num_agent)/$(printf %04d $seed)/out" &
		#python3 run_this.py -number_agent 4 -random_seed "$seed" > "./NUM_AGENT04/0001/out" &
	done
	wait
done


