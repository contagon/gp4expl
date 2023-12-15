# Working Reacher w/ GP
# TODO: I think this'll work with 400 inducing points and 1500 batch size
for i in 1 2 3; do
    for explore in -1 3 6 ; do
        python gp4expl/scripts/run.py --exp_name gp_explore${explore}_seed${i} --env_name reacher --add_sl_noise --video_log_freq -1 \
            --num_agent_train_steps_per_iter 1 --batch_size_initial 2000 --batch_size 2000 --train_batch_size 2000 --n_iter 10 \
            --mpc_horizon 10 --mpc_action_sampling_strategy 'random' \
            --num_inducing 500 --num_exploration_iterations ${explore} --dynamics_model gp  --random-target --seed ${i}
    done

    # NN
    python gp4expl/scripts/run.py --exp_name nn_seed${i} --env_name reacher --add_sl_noise --video_log_freq -1 \
        --num_agent_train_steps_per_iter 1000 --batch_size_initial 2000 --batch_size 2000 --train_batch_size 512 --n_iter 10 \
        --mpc_horizon 10 --mpc_action_sampling_strategy 'random' \
        --dynamics_model nn --seed ${i} --random-target
done