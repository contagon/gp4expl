# Working Reacher w/ GP
for BATCH in 200; do
    echo "BATCH ${BATCH}"
    python gp4expl/scripts/run.py --exp_name reacher_random_batch_${BATCH} --env_name reacher --add_sl_noise --video_log_freq -1 \
        --num_agent_train_steps_per_iter 1000 --batch_size_initial ${BATCH} --batch_size ${BATCH} --train_batch_size ${BATCH} --n_iter 20 \
        --mpc_horizon 10 --mpc_action_sampling_strategy 'random' \
        --num_inducing 500 --num_exploration_iterations 10 --dynamics_model nn  --random-target
done

# for BATCH in 200; do
#     echo "BATCH ${BATCH}"
#     python gp4expl/scripts/run.py --exp_name reacher_random_batch_${BATCH} --env_name reacher --add_sl_noise --video_log_freq -1 \
#         --num_agent_train_steps_per_iter 1 --batch_size_initial ${BATCH} --batch_size ${BATCH} --train_batch_size ${BATCH} --n_iter 20 \
#         --mpc_horizon 10 --mpc_action_sampling_strategy 'random' \
#         --num_inducing 500 --num_exploration_iterations 10 --dynamics_model gp  --random-target
# done