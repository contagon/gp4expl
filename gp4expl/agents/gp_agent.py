from .base_agent import BaseAgent
from gp4expl.models.gp_model import GPModel
from gp4expl.policies.MPC_policy import MPCPolicy
from gp4expl.infrastructure.replay_buffer import ReplayBuffer
from gp4expl.infrastructure.utils import *


class GPAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(GPAgent, self).__init__()

        self.env = env.unwrapped
        self.agent_params = agent_params
        self.ensemble_size = self.agent_params["ensemble_size"]
        self.ensemble_size = 1

        self.dyn_models = []
        for i in range(self.ensemble_size):
            model = GPModel(
                self.agent_params["ac_dim"],
                self.agent_params["ob_dim"],
                self.agent_params["num_inducing"],
            )
            self.dyn_models.append(model)

        self.explore_reward = lambda obs, action: self.dyn_models[0].get_reward(
            obs, action, self.data_statistics
        )

        self.actor = MPCPolicy(
            self.env,
            ac_dim=self.agent_params["ac_dim"],
            dyn_models=self.dyn_models,
            horizon=self.agent_params["mpc_horizon"],
            N=self.agent_params["mpc_num_action_sequences"],
            sample_strategy=self.agent_params["mpc_action_sampling_strategy"],
            cem_iterations=self.agent_params["cem_iterations"],
            cem_num_elites=self.agent_params["cem_num_elites"],
            cem_alpha=self.agent_params["cem_alpha"],
        )
        self.actor.reward_fun = self.explore_reward

        self.replay_buffer = ReplayBuffer()

        self.iteration_count = 0
        self.num_exploration_iterations = -1

        if self.num_exploration_iterations < 0:
            self.actor.reward_fun = self.env.get_reward

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensemble is trained on a different random batch of size batch_size
        if self.iteration_count > self.num_exploration_iterations:
            print("Switching rewards")
            self.actor.reward_fun = self.env.get_reward

        losses = []
        num_data = ob_no.shape[0]
        num_data_per_env = int(num_data / self.ensemble_size)

        for i in range(self.ensemble_size):
            # select which datapoints to use for this model of the ensemble
            # you might find the num_data_per_env variable defined above useful

            observations = ob_no[i * num_data_per_env : (i + 1) * num_data_per_env]
            actions = ac_na[i * num_data_per_env : (i + 1) * num_data_per_env]
            next_observations = next_ob_no[
                i * num_data_per_env : (i + 1) * num_data_per_env
            ]

            # use datapoints to update one of the dyn_models
            model = self.dyn_models[i]
            log = model.update(
                observations, actions, next_observations, self.data_statistics
            )

            loss = log["Training Loss"]
            losses.append(loss)

        avg_loss = np.mean(losses)

        self.iteration_count += 1

        return {
            "Training Loss": avg_loss,
        }

    def add_to_replay_buffer(self, paths, add_sl_noise=False):
        # add data to replay buffer
        self.replay_buffer.add_rollouts(paths, noised=add_sl_noise)

        # get updated mean/std of the data in our replay buffer
        self.data_statistics = {
            "obs_mean": np.mean(self.replay_buffer.obs, axis=0),
            "obs_std": np.std(self.replay_buffer.obs, axis=0),
            "acs_mean": np.mean(self.replay_buffer.acs, axis=0),
            "acs_std": np.std(self.replay_buffer.acs, axis=0),
            "delta_mean": np.mean(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0
            ),
            "delta_std": np.std(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0
            ),
        }

        # update the actor's data_statistics too, so actor.get_action can be calculated correctly
        self.actor.data_statistics = self.data_statistics

    def sample(self, batch_size):
        # NOTE: sampling batch_size * ensemble_size,
        # so each model in our ensemble can get trained on batch_size data
        return self.replay_buffer.sample_random_data(batch_size * self.ensemble_size)
