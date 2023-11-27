import numpy as np

from gym.envs.mujoco.inverted_double_pendulum_v4 import InvertedDoublePendulumEnv

class MyInvertedDoublePendulum(InvertedDoublePendulumEnv):
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
        p, sin1, sin2, cos1, cos2, v, v1, v2, _, _, _ = observations.T

        t1 = np.arctan2(sin1, cos1)
        t2 = np.arctan2(sin2, cos2)

        x = r*sin1 + r*np.sin(t1 + t2)
        y = r*cos1 + r*np.cos(t1 + t2)

        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10

        reward = alive_bonus - dist_penalty - vel_penalty
        terminated = y <= 1.0

        if not batch_mode:
            reward = reward[0]
            terminated = terminated[0]

        return reward, terminated