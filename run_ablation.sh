# Working Reacher w/ GP
for EXPLORE in 1 3 5; do
    python gp4expl/scripts/run.py --exp_name reacher_random_explore_${EXPLORE} --env_name reacher --add_sl_noise --video_log_freq -1 \
        --num_agent_train_steps_per_iter 1 --batch_size_initial 2000 --batch_size 2000 --train_batch_size 2000 --n_iter 15 \
        --mpc_horizon 10 --mpc_action_sampling_strategy 'random' \
        --num_inducing 500 --num_exploration_iterations ${EXPLORE} --dynamics_model gp  --random-target
done

for POINTS in 100 500; do
    python gp4expl/scripts/run.py --exp_name reacher_random_points_${POINTS} --env_name reacher --add_sl_noise --video_log_freq -1 \
        --num_agent_train_steps_per_iter 1 --batch_size_initial 2000 --batch_size 2000 --train_batch_size 2000 --n_iter 15 \
        --mpc_horizon 10 --mpc_action_sampling_strategy 'random' \
        --num_inducing ${POINTS} --num_exploration_iterations 3 --dynamics_model gp  --random-target
done