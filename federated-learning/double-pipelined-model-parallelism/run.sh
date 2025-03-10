#!/bin/bash
cd "$(dirname "$0")" || return

# Performs pytorch.distributed operations that are not directly supported by MPS backend
# (Metal Performance Shaders, i.e., macOS-specific backend) on the CPU instead.
# 'c10d::allreduce_' is one such unsupported operation
export PYTORCH_ENABLE_MPS_FALLBACK=1

total_model_ranks=3
total_stage_ranks=2

for ((model_rank=0; model_rank<total_model_ranks; model_rank=model_rank+1))
do
    for ((stage_rank=0; stage_rank<total_stage_ranks; stage_rank=stage_rank+1))
    do
        log_file="out/client-$model_rank-$stage_rank.log"

        touch "$log_file"
        (sleep 1; python -u "src/" $model_rank $stage_rank $total_stage_ranks >"$log_file") &
        pids+=($!)
    done
done

# Follow the output of the log files
tail -f out/client-*.log &

# Give the user the option to kill the processes
echo "Press [Enter] to stop the processes..."
read -r

# Kill the background processes
for pid in "${pids[@]}"
do
    echo "killing child processes of process with PID $pid"
    pkill -P "$pid"

    echo "killing process with PID $pid"
    kill -9 "$pid"
done

# Kill the tail process
kill %tail
