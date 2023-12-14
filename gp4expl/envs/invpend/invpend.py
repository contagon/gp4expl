import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv
from gym.spaces import Box


class MyInvertedPendulum(InvertedPendulumEnv):
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            "inverted_pendulum.xml",
            2,
            observation_space=observation_space,
            **kwargs,
        )

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward, terminated = self.get_reward(ob, a)
        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, False, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qpos[1] += 0.2
        # qpos[1] = np.sign(qpos[1]) * np.abs(qpos[1]) ** (1 / 4)
        # print(qpos[1])
        # qpos = np.zeros(2)
        # qpos[0] = self.np_random.uniform(low=-0.01, high=0.01)
        # qpos[1] = self.np_random.uniform(low=0.2, high=0.3)
        qpos[1] = qpos[1] if self.np_random.random() < 0.5 else -qpos[1]
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

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

        going = np.logical_and(
            np.isfinite(observations).all(axis=-1), np.abs(observations[:, 1]) < 0.5
        )
        terminated = ~going

        # Reward if we're still standing
        # reward = (~terminated).astype(np.int32)

        # Reward based on how upright we still are
        reward = (
            np.full_like(terminated, 1)
            - np.abs(observations[:, 1])
            - 1 * np.abs(observations[:, 0])
        )
        # reward = (~terminated).astype(np.float32) - np.abs(observations[:,1])

        if not batch_mode:
            reward = reward[0]
            terminated = terminated[0]

        return reward, terminated
