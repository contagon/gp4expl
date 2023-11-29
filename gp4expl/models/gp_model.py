import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn.preprocessing import StandardScaler


class GPModel:
    def __init__(self, ac_dim, ob_dim):
        super(GPModel, self).__init__()

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim

        self.X = np.zeros((0, ac_dim + ob_dim))
        self.y = np.zeros((0, ob_dim))
        self.kernel = C() * RBF() + W()
        self.n_restarts_optimizer = 10
        self.scalar = StandardScaler()
        self.delta_gp = None

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
        # predicted change in obs
        concatenated_input = np.hstack([obs_unnormalized, acs_unnormalized])

        # DONE(Q1) compute delta_pred_normalized and next_obs_pred
        # Hint: as described in the PDF, the output of the network is the
        # *normalized change* in state, i.e. normalized(s_t+1 - s_t).
        X = self.scalar.transform(concatenated_input)
        next_obs_pred = self.delta_gp.predict(X) + obs_unnormalized

        return next_obs_pred, None

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

        # predicted change in obs
        concatenated_input = np.hstack([observations, actions])
        target = next_observations - observations

        self.X = np.vstack((self.X, concatenated_input))
        self.y = np.vstack((self.y, target))

        self.delta_gp = GaussianProcessRegressor(
            self.kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=0,
            copy_X_train=True,
            normalize_y=True,
        )

        X = self.scalar.fit_transform(self.X)
        self.delta_gp.fit(X, self.y)

        self.kernel = self.delta_gp.kernel_
        self.n_restarts_optimizer = 2

        return {
            "Training Loss": 0,
        }
