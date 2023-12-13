# Working Reacher w/ GP
python gp4expl/scripts/run.py --exp_name inv_nn --env_name inverted-pendulum --add_sl_noise --video_log_freq -1 \
    --num_agent_train_steps_per_iter 1000 --batch_size_initial 5000 --batch_size 2000 --train_batch_size 512 --n_iter 15 \
    --mpc_horizon 10 --mpc_num_action_sequences 1000 --mpc_action_sampling_strategy 'cem' \
    --dynamics_model nn

# python gp4expl/scripts/run.py --exp_name reacher_set_expl --env_name reacher --add_sl_noise --video_log_freq -1 \
#     --num_agent_train_steps_per_iter 1 --batch_size_initial 2000 --batch_size 2000 --train_batch_size 2000 --n_iter 15 \
#     --mpc_horizon 10 --mpc_action_sampling_strategy 'random' \
#     --num_inducing 500 --num_exploration_iterations 3 --dynamics_model gp

# python gp4expl/scripts/run.py --exp_name reacher_set_gp --env_name reacher --add_sl_noise --video_log_freq -1 \
#     --num_agent_train_steps_per_iter 1 --batch_size_initial 2000 --batch_size 2000 --train_batch_size 2000 --n_iter 15 \
#     --mpc_horizon 10 --mpc_action_sampling_strategy 'random' \
#     --num_inducing 500 --num_exploration_iterations -1 --dynamics_model gp

# # With a randomized target
# python gp4expl/scripts/run.py --exp_name reacher_random_nn --env_name reacher --add_sl_noise --video_log_freq -1 \
#     --num_agent_train_steps_per_iter 1000 --batch_size_initial 2000 --batch_size 2000 --train_batch_size 512 --n_iter 15 \
#     --mpc_horizon 10 --mpc_action_sampling_strategy 'random' \
#     --dynamics_model nn --random-target

# python gp4expl/scripts/run.py --exp_name reacher_random_expl --env_name reacher --add_sl_noise --video_log_freq -1 \
#     --num_agent_train_steps_per_iter 1 --batch_size_initial 2000 --batch_size 2000 --train_batch_size 2000 --n_iter 15 \
#     --mpc_horizon 10 --mpc_action_sampling_strategy 'random' \
#     --num_inducing 500 --num_exploration_iterations 3 --dynamics_model gp  --random-target
for POINTS in 600; do
    python gp4expl/scripts/run.py --exp_name inv_gp_${POINTS} --env_name inverted-pendulum --add_sl_noise --video_log_freq -1 \
        --num_agent_train_steps_per_iter 1 --batch_size_initial 1000 --batch_size 1000 --train_batch_size 1000 --n_iter 15 \
        --mpc_horizon 10 --mpc_num_action_sequences 1000 --mpc_action_sampling_strategy 'cem' \
        --num_inducing ${POINTS} --num_exploration_iterations -1 --dynamics_model gp
done