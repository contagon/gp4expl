# # Working IP w/ GP
# python gp4expl/scripts/run.py --exp_name mine_explore0 --env_name inverted-pendulum --add_sl_noise --video_log_freq -1 \
#     --num_agent_train_steps_per_iter 1 --batch_size_initial 2000 --batch_size 2000 --train_batch_size 2000 --n_iter 30 \
#     --mpc_horizon 3 --mpc_action_sampling_strategy 'cem' --cem_iterations 2 --mpc_num_action_sequences 500 \
#     --num_inducing 50 --dynamics_model gp

# python gp4expl/scripts/run.py --exp_name mine_explore5 --env_name inverted-pendulum --add_sl_noise --video_log_freq -1 \
#     --num_agent_train_steps_per_iter 1 --batch_size_initial 200 --batch_size 200 --train_batch_size 200 --n_iter 30 \
#     --mpc_horizon 5 --mpc_action_sampling_strategy 'cem' --cem_iterations 2 --mpc_num_action_sequences 100 \
#     --num_inducing 20 --dynamics_model gp --num_exploration_iterations 5

# python gp4expl/scripts/run.py --exp_name mine_explore10 --env_name inverted-pendulum --add_sl_noise --video_log_freq -1 \
#     --num_agent_train_steps_per_iter 1 --batch_size_initial 200 --batch_size 200 --train_batch_size 200 --n_iter 30 \
#     --mpc_horizon 5 --mpc_action_sampling_strategy 'cem' --cem_iterations 2 --mpc_num_action_sequences 100 \
#     --num_inducing 20 --dynamics_model gp --num_exploration_iterations 10


# python gp4expl/scripts/run.py --exp_name mine_explore15 --env_name inverted-pendulum --add_sl_noise --video_log_freq -1 \
#     --num_agent_train_steps_per_iter 1 --batch_size_initial 200 --batch_size 200 --train_batch_size 200 --n_iter 30 \
#     --mpc_horizon 5 --mpc_action_sampling_strategy 'cem' --cem_iterations 2 --mpc_num_action_sequences 100 \
#     --num_inducing 20 --dynamics_model gp --num_exploration_iterations 15

# python gp4expl/scripts/run.py --exp_name mine_explore20 --env_name inverted-pendulum --add_sl_noise --video_log_freq -1 \
#     --num_agent_train_steps_per_iter 1 --batch_size_initial 200 --batch_size 200 --train_batch_size 200 --n_iter 30 \
#     --mpc_horizon 5 --mpc_action_sampling_strategy 'cem' --cem_iterations 2 --mpc_num_action_sequences 100 \
#     --num_inducing 20 --dynamics_model gp --num_exploration_iterations 20

# # # NN
python gp4expl/scripts/run.py --exp_name nn --env_name inverted-pendulum --add_sl_noise --video_log_freq -1 \
    --num_agent_train_steps_per_iter 50 --batch_size_initial 200 --batch_size 200 --train_batch_size 200 --n_iter 30 \
    --mpc_horizon 2 --mpc_action_sampling_strategy 'cem' --cem_iterations 2 --mpc_num_action_sequences 100 \
    --num_inducing 20 --dynamics_model nn --seed 10