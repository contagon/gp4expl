import os
import time

from gp4expl.infrastructure.rl_trainer import RL_Trainer
from gp4expl.agents.mb_agent import MBAgent
from gp4expl.agents.gp_agent import GPAgent

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class MB_Trainer(object):
    def __init__(self, params):
        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            "ensemble_size": params["ensemble_size"],
            "n_layers": params["n_layers"],
            "size": params["size"],
            "learning_rate": params["learning_rate"],
            "num_inducing": params["num_inducing"],
            "num_exploration_iterations": params["num_exploration_iterations"],
        }

        train_args = {
            "num_agent_train_steps_per_iter": params["num_agent_train_steps_per_iter"],
        }

        controller_args = {
            "mpc_horizon": params["mpc_horizon"],
            "mpc_num_action_sequences": params["mpc_num_action_sequences"],
            "mpc_action_sampling_strategy": params["mpc_action_sampling_strategy"],
            "cem_iterations": params["cem_iterations"],
            "cem_num_elites": params["cem_num_elites"],
            "cem_alpha": params["cem_alpha"],
        }

        agent_params = {**computation_graph_args, **train_args, **controller_args}

        self.params = params
        if params["dynamics_model"] == "gp":
            self.params["agent_class"] = GPAgent
        elif params["dynamics_model"] == "nn":
            self.params["agent_class"] = MBAgent

        self.params["agent_params"] = agent_params

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.params["n_iter"],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name", type=str
    )  # reacher-hw4_part1-v0, ant-hw4_part1-v0, cheetah-hw4_part1-v0, obstacles-hw4_part1-v0
    parser.add_argument("--ep_len", type=int, default=200)
    parser.add_argument("--exp_name", type=str, default="todo")
    parser.add_argument("--n_iter", "-n", type=int, default=20)
    parser.add_argument("--random-target", action="store_true")

    parser.add_argument(
        "--dynamics_model", type=str, default="gp", choices=["gp", "nn"]
    )

    # GP Arguments
    parser.add_argument("--num_inducing", type=int, default=100)
    parser.add_argument("--num_exploration_iterations", type=int, default=-1)

    # NN Arguments
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--size", "-s", type=int, default=250)

    # MPC Arguments
    parser.add_argument("--ensemble_size", "-e", type=int, default=3)
    parser.add_argument("--mpc_horizon", type=int, default=10)
    parser.add_argument("--mpc_num_action_sequences", type=int, default=1000)
    parser.add_argument("--mpc_action_sampling_strategy", type=str, default="random")
    parser.add_argument("--cem_iterations", type=int, default=4)
    parser.add_argument("--cem_num_elites", type=int, default=5)
    parser.add_argument("--cem_alpha", type=float, default=1)

    # RL Arguments
    parser.add_argument("--add_sl_noise", "-noise", action="store_true")
    parser.add_argument("--num_agent_train_steps_per_iter", type=int, default=1000)
    # (random) steps collected on 1st iteration (put into replay buffer)
    parser.add_argument("--batch_size_initial", type=int, default=20000)
    # steps collected per train iteration (put into replay buffer)
    parser.add_argument("--batch_size", "-b", type=int, default=8000)
    ##steps used per gradient step (used for training)
    parser.add_argument("--train_batch_size", "-tb", type=int, default=512)
    # steps collected per eval iteration
    parser.add_argument("--eval_batch_size", "-eb", type=int, default=400)

    # Misc Arguments
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)  # -1 to disable
    parser.add_argument("--scalar_log_freq", type=int, default=1)  # -1 to disable
    parser.add_argument("--save_params", action="store_true")
    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    # HARDCODE EPISODE LENGTHS FOR THE ENVS USED IN THIS MB ASSIGNMENT
    if params["env_name"] == "reacher-hw4_part1-v0":
        params["ep_len"] = 200
    if params["env_name"] == "cheetah-hw4_part1-v0":
        params["ep_len"] = 500
    if params["env_name"] == "obstacles-hw4_part1-v0":
        params["ep_len"] = 100
    if params["env_name"] == "inverted-pendulum":
        params["ep_len"] = 100
    if params["env_name"] == "double-inverted-pendulum":
        params["ep_len"] = 200

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = (
        args.exp_name + "_" + args.env_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(data_path, logdir)
    params["logdir"] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\nLOGGING TO: ", logdir, "\n")

    ###################
    ### RUN TRAINING
    ###################

    trainer = MB_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
