#!/bin/bash

cd "$pwd"
#mkdir -p Reward
#mkdir -p Saved_model
#mkdir -p Travel_distance
for number in 01 02 03 04 05 06 07 08 09 10 11 12
do
        mkdir -p "NUM_AGENT$number"
done

array=(*/)
for dir in "${array[@]}"
do
        echo "$dir"
        cd "$dir"
        #mkdir -p reward_output
        #mkdir -p Saved_model
        #mkdir -p distance_output
        for number in $(seq -f  "%04g" 1 100)
        do
                mkdir -p  "$number"
        done
        cd ..
done

#mkdir -p Reward
#mkdir -p Saved_model
#mkdir -p Travel_distance
