import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.envs.mujoco.reacher_v4 import ReacherEnv
from gym.spaces import Box


class MyReacherEnv(ReacherEnv):
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64)
        MujocoEnv.__init__(
            self, "reacher.xml", 2, observation_space=observation_space, **kwargs
        )

    def _get_obs(self):
        theta = self.data.qpos.flat[:2]
        return np.concatenate(
            [
                theta,
                self.data.qpos.flat[2:],
                self.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        )

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        reward = self.get_reward(ob, a)[0]
        return (
            ob,
            reward,
            False,
            False,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
        )

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

        vec = observations[:, -3:]
        reward_dist = -np.linalg.norm(vec, axis=1)
        reward_ctrl = -np.square(actions).sum(axis=1)
        reward = reward_dist + 0.1 * reward_ctrl

        terminated = np.full_like(reward, False, dtype=np.bool)

        if not batch_mode:
            reward = reward[0]
            terminated = terminated[0]

        return reward, terminated
