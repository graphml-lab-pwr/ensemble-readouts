#!/bin/bash

# Runs parallel experiments through tmux session, constraints maximum number of parallel jobs

set -e

eval "$(conda shell.bash hook)"
conda activate ensemble-readouts

config_path=${config_path:-""}
config_list=${config_list:-""}
config_list_names=${config_list_names:-"config_names_light"}
num_jobs=${num_jobs:-"1"}
accelerator=${accelerator:-"cpu"}
devices=${devices:-""}
code=${code:-"experiments/scripts/train_gnn_with_reps.py"}
output_dir=${output_dir:-"shell_outputs"}


# parsing positional arguments
while [ $# -gt 0 ]; do

  if [[ $1 == *"--"* ]]; then
    param="${1/--/}"
    param=$(echo "$param" | tr '-' '_')
    declare "$param"="$2"
  fi

  shift
done

mkdir -p "$output_dir"

filename=$(basename -- "$config_path")
config_name="${filename%.*}"
dataset_name="$( cut -d '_' -f 2- <<< "$config_name" )"

# check obligatory args were passed
if [[ "$config_path" == "" ]]; then
  echo "You need to pass --config-path"
  exit 1
fi
if [[ "$config_list" == "" ]]; then
  echo "You need to pass --config-list"
  exit 1
fi

config_keys=()
# reads file skipping commented lines
while read -r config_option; do
  config_keys+=("$config_option")
done < <(shyaml get-values "$config_list_names" <"$config_list")

# if no configuration options -> stop the program
if ! ((${#config_keys[@]} > 0)); then
  echo "Configuration list is empty"
  exit 1
fi

commands=()
for ((i = 0; i < ${#config_keys[@]}; i++)); do
  cfg_key="${config_keys[i]}"
  commands+=("PYTHONPATH=. WANDB_MODE=offline python $code --config-path $config_path --config-key $cfg_key --accelerator $accelerator ${devices:+"--devices $devices"}")
done

# in case of problems kill all started background jobs to avoid dangling processes
trap "jobs -p | xargs kill -s SIGINT" SIGINT ERR

### Runs jobs constraining number of parallel ones ###
for i in "${!commands[@]}"; do
  # sleep until some script finished
  while [[ $(jobs -p | wc -l) -ge $num_jobs ]]; do
    sleep 30
  done

  # print and run script
  echo "${commands[i]}"
  eval "${commands[i]} &> $output_dir/$(date '+%Y_%m_%d')_${dataset_name}_${config_keys[i]}.out" &

done
wait
