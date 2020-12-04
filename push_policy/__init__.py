from gym.envs.registration import register
register(
    id='PushNav-v0',
    entry_point='push_policy.envs:PushNavEnv'
)