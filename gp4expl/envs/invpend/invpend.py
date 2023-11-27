import numpy as np

from gym.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv

class MyInvertedPendulum(InvertedPendulumEnv):
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

        terminated = np.logical_and(
            np.isfinite(observations).all(axis=-1),
            np.abs(observations[:,1]) > 0.2
        )

        # Reward if we're still standing
        # reward = (~terminated).astype(np.int32)
        
        # Reward based on how upright we still are
        reward = np.full_like(terminated, 1) - np.abs(observations[:,1])


        if not batch_mode:
            reward = reward[0]
            terminated = terminated[0]

        return reward, terminated