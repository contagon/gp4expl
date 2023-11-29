import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

from pathlib import Path


class MyInvertedDoublePendulum(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, **kwargs):
        model_file = Path(__file__).parent / "inverted_double_pendulum_wide.xml"
        observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            str(model_file),
            5,
            observation_space=observation_space,
            **kwargs,
        )
        utils.EzPickle.__init__(self, **kwargs)

    def step(self, action):
        self.do_simulation(action.flatten(), self.frame_skip)
        ob = self._get_obs()
        r, terminated = self.get_reward(ob, action)
        self.renderer.render_step()
        return ob, r, terminated, False, {}

    def reset_model(self):
        # Start at the lowest point
        self.set_state(
            np.array([0, np.pi, 0])
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1,
        )
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]

    # ------------------------- My custom adjustments ------------------------- #
    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos[:1],  # cart x pos
                self.data.qpos[1:],  # link angles
                np.clip(self.data.qvel, -10, 10),  # cart & angle velocity
            ]
        ).ravel()

    def theta2xy(self, observations):
        r = 0.6
        p, t1, t2, v, v1, v2 = observations.T

        x = r * np.sin(t1) + r * np.sin(t1 + t2)
        y = r * np.cos(t1) + r * np.cos(t1 + t2)
        return x, y

    def get_reward(self, observations, actions):
        """get reward/s of given (observations, actions) datapoint or datapoints

        Args:
            observations: (batchsize, obs_dim) or (obs_dim,)
            actions: (batchsize, ac_dim) or (ac_dim,)

        Return:
            r_total: reward of this (o,a) pair, dimension is (batchsize,1) or (1,)
            done: True if env reaches terminal state, dimension is (batchsize,1) or (1,)
        """

        # TODO: Adjust reward to different tasks
        # initialize and reshape as needed, for batch mode
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, axis=0)
            actions = np.expand_dims(actions, axis=0)
            batch_mode = False
        else:
            batch_mode = True

        p, t1, t2, v, v1, v2 = observations.T
        x, y = self.theta2xy(observations)

        dist_penalty = x**2 + (y - 1.2) ** 2
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10

        reward = alive_bonus - dist_penalty - vel_penalty
        terminated = np.logical_and(
            ~np.isfinite(observations).all(axis=-1),
            (observations[:, 3] < 5),
            (observations[:, 4:] < 10).all(axis=-1),
        )

        if not batch_mode:
            reward = reward[0]
            terminated = terminated[0]

        return reward, terminated
