import numpy as np
from gp4expl.models.base_model import BaseModel
from gp4expl.infrastructure.utils import normalize, unnormalize
from gp4expl.infrastructure import pytorch_util as ptu
from goppy import OnlineGP, SquaredExponentialKernel


class GPModel:
    def __init__(self, ac_dim, ob_dim):
        super(GPModel, self).__init__()

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim

        self.delta_gp = OnlineGP(SquaredExponentialKernel(0.3), noise_var=0.1)

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
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.acs_mean = acs_mean
        self.acs_std = acs_std
        self.delta_mean = delta_mean
        self.delta_std = delta_std

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
        concatenated_input = np.hstack([obs_normalized, acs_normalized])

        # DONE(Q1) compute delta_pred_normalized and next_obs_pred
        # Hint: as described in the PDF, the output of the network is the
        # *normalized change* in state, i.e. normalized(s_t+1 - s_t).
        delta_pred_normalized = self.delta_gp.predict(concatenated_input, what="mean")
        next_obs_pred = obs_unnormalized + (
            delta_pred_normalized["mean"] * self.delta_std + self.delta_mean
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
        prediction, delta = self.forward(obs, acs, **data_statistics)
        return prediction

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
        self.update_statistics(**data_statistics)

        obs_normalized = (observations - self.obs_mean) / self.obs_std
        acs_normalized = (actions - self.acs_mean) / self.acs_std
        # predicted change in obs
        concatenated_input = np.hstack([obs_normalized, acs_normalized])

        obs_next_normalized = (next_observations - self.obs_mean) / self.obs_std

        self.delta_gp.add(concatenated_input, obs_next_normalized)
        print("TRAINING", self.delta_gp.x_train.shape)

        return {
            "Training Loss": 0,
        }
