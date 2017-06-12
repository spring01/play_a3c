#!/bin/bash

python play_dist.py \
    --ps_hosts=localhost:2221 \
    --worker_hosts=localhost:2222,localhost:2223,localhost:2224,localhost:2225 \
    --job_name=ps --task_index=0 &

sleep 5

python play_dist.py \
    --ps_hosts=localhost:2221 \
    --worker_hosts=localhost:2222,localhost:2223,localhost:2224,localhost:2225 \
    --job_name=worker --task_index=0 &
python play_dist.py \
    --ps_hosts=localhost:2221 \
    --worker_hosts=localhost:2222,localhost:2223,localhost:2224,localhost:2225 \
    --job_name=worker --task_index=1 &
python play_dist.py \
    --ps_hosts=localhost:2221 \
    --worker_hosts=localhost:2222,localhost:2223,localhost:2224,localhost:2225 \
    --job_name=worker --task_index=2 &
python play_dist.py \
    --ps_hosts=localhost:2221 \
    --worker_hosts=localhost:2222,localhost:2223,localhost:2224,localhost:2225 \
    --job_name=worker --task_index=3 &
