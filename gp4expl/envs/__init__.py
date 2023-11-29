from gym.envs.registration import register


def register_envs():
    register(
        id="cheetah-hw4_part1-v0",
        entry_point="gp4expl.envs.cheetah:HalfCheetahEnv",
        max_episode_steps=1000,
    )
    register(
        id="obstacles-hw4_part1-v0",
        entry_point="gp4expl.envs.obstacles:Obstacles",
        max_episode_steps=500,
    )
    register(
        id="reacher-hw4_part1-v0",
        entry_point="gp4expl.envs.reacher:Reacher7DOFEnv",
        max_episode_steps=500,
    )
    register(
        id="inverted-pendulum",
        entry_point="gp4expl.envs.invpend:MyInvertedPendulum",
        max_episode_steps=200,
    )
    register(
        id="double-inverted-pendulum",
        entry_point="gp4expl.envs.invdoublepend:MyInvertedDoublePendulum",
        max_episode_steps=200,
    )
