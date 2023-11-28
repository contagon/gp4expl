import numpy as np

from gym.envs.mujoco.inverted_double_pendulum_v4 import InvertedDoublePendulumEnv
from gym.spaces import Box
from gym.envs.mujoco import MujocoEnv
from gym import utils

class MyInvertedDoublePendulum(InvertedDoublePendulumEnv):
    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            "inverted_double_pendulum.xml",
            5,
            observation_space=observation_space,
            **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos[:1],  # cart x pos
                self.data.qpos[1:],  # link angles
                np.clip(self.data.qvel, -10, 10), # cart & angle velocity
            ]
        ).ravel()

    def get_reward(self, observations, actions):
        """get reward/s of given (observations, actions) datapoint or datapoints

        Args:
            observations: (batchsize, obs_dim) or (obs_dim,)
            actions: (batchsize, ac_dim) or (ac_dim,)

        Return:
            r_total: reward of this (o,a) pair, dimension is (batchsize,1) or (1,)
            done: True if env reaches terminal state, dimension is (batchsize,1) or (1,)
        """

        # initialize and reshape as needed, for batch mode
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, axis=0)
            actions = np.expand_dims(actions, axis=0)
            batch_mode = False
        else:
            batch_mode = True

        r = 0.6
        p, t1, t2, v, v1, v2 = observations.T

        x = r*np.sin(t1) + r*np.sin(t1 + t2)
        y = r*np.cos(t1) + r*np.cos(t1 + t2)

        dist_penalty = x**2 + (y - 1.2) ** 2
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10

        reward = alive_bonus - dist_penalty - vel_penalty
        terminated = y <= 1.0

        if not batch_mode:
            reward = reward[0]
            terminated = terminated[0]

        return reward, terminated