# # Working Reacher w/ GP
# python gp4expl/scripts/run.py --exp_name reacher_set_nn --env_name reacher --add_sl_noise --video_log_freq -1 \
#     --num_agent_train_steps_per_iter 1000 --batch_size_initial 2000 --batch_size 2000 --train_batch_size 512 --n_iter 15 \
#     --mpc_horizon 10 --mpc_action_sampling_strategy 'random' \
#     --dynamics_model nn

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

python gp4expl/scripts/run.py --exp_name reacher_random_gp_new_induce --env_name reacher --add_sl_noise --video_log_freq -1 \
    --num_agent_train_steps_per_iter 1 --batch_size_initial 2000 --batch_size 2000 --train_batch_size 2000 --n_iter 15 \
    --mpc_horizon 10 --mpc_action_sampling_strategy 'random' \
    --num_inducing 500 --num_exploration_iterations -1 --dynamics_model gp  --random-target
