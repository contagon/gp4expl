# Working IP w/ GP
# python gp4expl/scripts/run.py --exp_name mine --env_name inverted-pendulum --add_sl_noise --video_log_freq -1 \
#     --num_agent_train_steps_per_iter 1 --batch_size_initial 500 --batch_size 500 --train_batch_size 500 --n_iter 15 \
#     --mpc_horizon 5 --mpc_action_sampling_strategy 'cem' --cem_iterations 2 --mpc_num_action_sequences 100 \
#     --num_inducing 10 --dynamics_model gp

# NN
python gp4expl/scripts/run.py --exp_name mine --env_name inverted-pendulum --add_sl_noise --video_log_freq -1 \
    --num_agent_train_steps_per_iter 1 --batch_size_initial 500 --batch_size 500 --train_batch_size 500 --n_iter 15 \
    --mpc_horizon 5 --mpc_action_sampling_strategy 'cem' --cem_iterations 2 --mpc_num_action_sequences 100 \
    --num_inducing 10 --dynamics_model nn

# Working IP w/ GP
# python gp4expl/scripts/run.py --exp_name mine --env_name inverted-pendulum --add_sl_noise --video_log_freq -1 \
#     --num_agent_train_steps_per_iter 1 --batch_size_initial 500 --batch_size 500 --train_batch_size 500 --n_iter 15 \
#     --mpc_horizon 5 --mpc_action_sampling_strategy 'cem' --cem_iterations 2 --mpc_num_action_sequences 100 \
#     --num_inducing 10 --dynamics_model gp
