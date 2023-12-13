import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv
from gym.spaces import Box

from pathlib import Path


class MyInvertedPendulum(InvertedPendulumEnv):
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        model_file = Path(__file__).parent / "invpend.xml"
        MujocoEnv.__init__(
            self,
            str(model_file),
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
        qpos = np.array([0, np.pi]) + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        state = self.data.qpos
        # state[1] = (state[1] + np.pi / 4) % (2 * np.pi) - np.pi / 4
        return np.concatenate([state, self.data.qvel]).ravel()

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

        terminated = ~np.isfinite(observations).all(axis=-1)

        # Reward if we're still standing
        # reward = (~terminated).astype(np.int32)

        # Reward based on how upright we still are
        angle = (observations[:, 1] + np.pi) % (2 * np.pi) - np.pi
        x = observations[:, 0]
        vel = observations[:, 3]
        reward = (
            np.full_like(terminated, 1)
            - np.abs(angle)
            - 0.05 * np.abs(vel)
            - 0.1 * np.abs(x)
        )
        # reward = (~terminated).astype(np.float32) - np.abs(observations[:,1])

        if not batch_mode:
            reward = reward[0]
            terminated = terminated[0]

        return reward, terminated
