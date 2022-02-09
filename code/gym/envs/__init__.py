from gym.envs.registration import registry, register, make, spec
# Database
# ----------------------------------------


for i in range(0, 10):
    train_env_name = 'PGsql-Train-Join-Job-v{}'.format(i)
    train_file_name = 'agents/queries/crossval_sens/job_queries_simple_crossval_{}_train.txt'.format(i)

    test_env_name = 'PGsql-Eval-Join-Job-v{}'.format(i)
    test_file_name = 'agents/queries/crossval_sens/job_queries_simple_crossval_{}_test.txt'.format(i)
    register(
        id=train_env_name,
        kwargs={'file_path': train_file_name},
        # entry_point='gym.envs.database2:Train_Join_Job'
        # entry_point='gym.envs.database2:Train_Join_Step_Reward'
        entry_point = 'gym.envs.database2:Train_Join_Step_Tree_Struct'
    )

    register(
        id=test_env_name, #id是调用所构建的环境的时候起的名字
        kwargs={'file_path':test_file_name},
        entry_point='gym.envs.database2:Test_Join_Step_Tree_Struct',#entry_point是环境文件所在的位置
        # max_episode_steps=20
    )



# Algorithmic
# ----------------------------------------

register(
    id='Copy-v0',
    entry_point='gym.envs.algorithmic:CopyEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='RepeatCopy-v0',
    entry_point='gym.envs.algorithmic:RepeatCopyEnv',
    max_episode_steps=200,
    reward_threshold=75.0,
)

register(
    id='ReversedAddition-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows' : 2},
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='ReversedAddition3-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows' : 3},
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='DuplicatedInput-v0',
    entry_point='gym.envs.algorithmic:DuplicatedInputEnv',
    max_episode_steps=200,
    reward_threshold=9.0,
)

register(
    id='Reverse-v0',
    entry_point='gym.envs.algorithmic:ReverseEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)
