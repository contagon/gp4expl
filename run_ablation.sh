# Working Reacher w/ GP
# for EXPLORE in 1 3 5; do
#     python gp4expl/scripts/run.py --exp_name reacher_random_explore_${EXPLORE} --env_name reacher --add_sl_noise --video_log_freq -1 \
#         --num_agent_train_steps_per_iter 1 --batch_size_initial 2000 --batch_size 2000 --train_batch_size 2000 --n_iter 15 \
#         --mpc_horizon 10 --mpc_action_sampling_strategy 'random' \
#         --num_inducing 500 --num_exploration_iterations ${EXPLORE} --dynamics_model gp  --random-target
# done

for POINTS in 300; do
    echo "POINTS ${POINTS}"
    python gp4expl/scripts/run.py --exp_name reacher_points_${POINTS} --env_name small-reacher --add_sl_noise --video_log_freq -1 \
        --num_agent_train_steps_per_iter 1 --batch_size_initial 1000 --batch_size 1000 --train_batch_size 1000 --n_iter 10 \
        --mpc_horizon 5 --mpc_action_sampling_strategy 'random' --mpc_num_action_sequences 3000 \
        --num_inducing ${POINTS} --num_exploration_iterations -1 --dynamics_model gp  --random-target
done

# for BATCH_SIZE in 200 400 600 1000; do
#     echo "BATCH SIZE ${BATCH_SIZE}"
#     python gp4expl/scripts/run.py --exp_name reacher_batch_${BATCH_SIZE} --env_name small-reacher --add_sl_noise --video_log_freq -1 \
#         --num_agent_train_steps_per_iter 1 --batch_size_initial ${BATCH_SIZE} --batch_size ${BATCH_SIZE} --train_batch_size ${BATCH_SIZE} --n_iter 10 \
#         --mpc_horizon 2 --mpc_action_sampling_strategy 'random' \
#         --num_inducing 300 --num_exploration_iterations -1 --dynamics_model gp
# done

# for MPC in 1 2 5; do
#     echo "MPC ${MPC}"
#     python gp4expl/scripts/run.py --exp_name reacher_mpc_${MPC} --env_name small-reacher --add_sl_noise --video_log_freq -1 \
#         --num_agent_train_steps_per_iter 1 --batch_size_initial 600 --batch_size 600 --train_batch_size 600 --n_iter 10 \
#         --mpc_horizon ${MPC} --mpc_action_sampling_strategy 'random' \
#         --num_inducing 300 --num_exploration_iterations -1 --dynamics_model gp
# done
