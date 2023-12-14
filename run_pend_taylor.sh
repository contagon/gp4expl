# # Working IP w/ GP
for i in 1 2 3 4 5; do
    for explore in -1 4 9 14; do
        python gp4expl/scripts/run.py --exp_name gp_explore${explore}_seed${i} --env_name inverted-pendulum --add_sl_noise --video_log_freq -1 \
            --num_agent_train_steps_per_iter 1 --batch_size_initial 200 --batch_size 200 --train_batch_size 200 --n_iter 20 \
            --mpc_horizon 2 --mpc_action_sampling_strategy 'cem' --cem_iterations 2 --mpc_num_action_sequences 100 \
            --num_inducing 20 --dynamics_model gp --num_exploration_iterations ${explore} --seed ${i}
    done

    # NN
    python gp4expl/scripts/run.py --exp_name nn_seed${i} --env_name inverted-pendulum --add_sl_noise --video_log_freq -1 \
        --num_agent_train_steps_per_iter 50 --batch_size_initial 200 --batch_size 200 --train_batch_size 200 --n_iter 20 \
        --mpc_horizon 2 --mpc_action_sampling_strategy 'cem' --cem_iterations 2 --mpc_num_action_sequences 100 \
        --dynamics_model nn --seed ${i}
done