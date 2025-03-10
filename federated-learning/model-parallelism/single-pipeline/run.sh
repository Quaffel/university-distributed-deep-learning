#!/bin/bash
cd "$(dirname "$0")" || return

get_log_file_for_rank() {
    rank=$1
    echo 
}

for ((rank=0; rank<3; rank=rank+1))
do
    log_file="out/client$rank.log"

    touch "$log_file"
    (sleep 1; python -u "src/" $rank>"$log_file") &
    pids+=($!)
done

# Follow the output of the log files
tail -f out/client*.log &

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
