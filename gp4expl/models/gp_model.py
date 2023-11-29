from torch import nn
import torch
from torch import optim
from gp4expl.models.base_model import BaseModel
from gp4expl.infrastructure.utils import normalize, unnormalize
from gp4expl.infrastructure import pytorch_util as ptu


class GPModel(nn.Module, BaseModel):
    def __init__(self):
        super(GPModel, self).__init__()

        self.obs_mean = None
        self.obs_std = None
        self.acs_mean = None
        self.acs_std = None
        self.delta_mean = None
        self.delta_std = None

    def update_statistics(
        self,
        obs_mean,
        obs_std,
        acs_mean,
        acs_std,
        delta_mean,
        delta_std,
    ):
        self.obs_mean = ptu.from_numpy(obs_mean)
        self.obs_std = ptu.from_numpy(obs_std)
        self.acs_mean = ptu.from_numpy(acs_mean)
        self.acs_std = ptu.from_numpy(acs_std)
        self.delta_mean = ptu.from_numpy(delta_mean)
        self.delta_std = ptu.from_numpy(delta_std)

    def forward(
        self,
        obs_unnormalized,
        acs_unnormalized,
        obs_mean,
        obs_std,
        acs_mean,
        acs_std,
        delta_mean,
        delta_std,
    ):
        """
        :param obs_unnormalized: Unnormalized observations
        :param acs_unnormalized: Unnormalized actions
        :param obs_mean: Mean of observations
        :param obs_std: Standard deviation of observations
        :param acs_mean: Mean of actions
        :param acs_std: Standard deviation of actions
        :param delta_mean: Mean of state difference `s_t+1 - s_t`.
        :param delta_std: Standard deviation of state difference `s_t+1 - s_t`.
        :return: tuple `(next_obs_pred, delta_pred_normalized)`
        This forward function should return a tuple of two items
            1. `next_obs_pred` which is the predicted `s_t+1`
            2. `delta_pred_normalized` which is the normalized (i.e. not
                unnormalized) output of the delta network. This is needed
        """
        self.update_statistics(
            obs_mean, obs_std, acs_mean, acs_std, delta_mean, delta_std
        )
        # normalize input data to mean 0, std 1
        obs_normalized = (obs_unnormalized - self.obs_mean) / self.obs_std
        acs_normalized = (acs_unnormalized - self.acs_mean) / self.acs_std
        # predicted change in obs
        concatenated_input = torch.cat([obs_normalized, acs_normalized], dim=1)

        # DONE(Q1) compute delta_pred_normalized and next_obs_pred
        # Hint: as described in the PDF, the output of the network is the
        # *normalized change* in state, i.e. normalized(s_t+1 - s_t).
        delta_pred_normalized = self.delta_network(concatenated_input)
        next_obs_pred = obs_unnormalized + (
            delta_pred_normalized * self.delta_std + self.delta_mean
        )
        return next_obs_pred, delta_pred_normalized

    def get_prediction(self, obs, acs, data_statistics):
        """
        :param obs: numpy array of observations (s_t)
        :param acs: numpy array of actions (a_t)
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return: a numpy array of the predicted next-states (s_t+1)
        """
        # DONE(Q1) get the predicted next-states (s_t+1) as a numpy array
        # Hint: `self(...)` returns a tuple, but you only need to use one of the
        # outputs.
        prediction, delta = self(
            ptu.from_numpy(obs), ptu.from_numpy(acs), **data_statistics
        )
        return ptu.to_numpy(prediction)

    def update(self, observations, actions, next_observations, data_statistics):
        """
        :param observations: numpy array of observations
        :param actions: numpy array of actions
        :param next_observations: numpy array of next observations
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return:
        """
        # DONE(Q1) compute the normalized target for the model.
        # Hint: you should use `data_statistics['delta_mean']` and
        # `data_statistics['delta_std']`, which keep track of the mean
        # and standard deviation of the model.
        target = (
            (next_observations - observations) - data_statistics["delta_mean"]
        ) / data_statistics["delta_std"]
        target = ptu.from_numpy(target)

        # DONE(Q1) compute the loss
        # Hint: `self(...)` returns a tuple, but you only need to use one of the
        # outputs.
        delta_pred = self(
            ptu.from_numpy(observations), ptu.from_numpy(actions), **data_statistics
        )[1]
        loss = self.loss(delta_pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Training Loss": ptu.to_numpy(loss),
        }