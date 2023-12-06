# Working IP w/ GP
python gp4expl/scripts/run.py --exp_name reacher --env_name reacher --add_sl_noise --video_log_freq -1 \
    --num_agent_train_steps_per_iter 1 --batch_size_initial 2000 --batch_size 2000 --train_batch_size 2000 --n_iter 15 \
    --mpc_horizon 10 --mpc_action_sampling_strategy 'random' \
    --num_inducing 1000 --dynamics_model gp

# python gp4expl/scripts/run.py --exp_name q4_reacher_ensemble1 --env_name reacher --ensemble_size 1 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random' --dynamics_model nn