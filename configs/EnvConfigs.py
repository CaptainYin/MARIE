from configs.Config import Config
from env.flatland.EnvCurriculum import EnvCurriculum, EnvCurriculumSample, EnvCurriculumPrioritizedSample


class EnvConfig(Config):
    def __init__(self):
        pass

    def create_env(self):
        pass


class StarCraftConfig(EnvConfig):
    def __init__(self, env_name, seed):
        self.env_name = env_name
        self.seed = seed

    def create_env(self):
        from env.starcraft.StarCraft import StarCraft

        return StarCraft(self.env_name, self.seed)


class SMAXConfig(EnvConfig):
    def __init__(self, env_name, seed, **kwargs):
        self.env_name = env_name
        self.seed = seed
        self.kwargs = kwargs

    def create_env(self):
        from env.smax.SMAX import SMAX

        return SMAX(self.env_name, self.seed, **self.kwargs)


class SMACv2Config(EnvConfig):
    def __init__(self, map_name, seed):
        self.map_name = map_name
        self.seed = seed

    def create_env(self):
        from env.smacv2.smacv2_env import SMACv2Env

        return SMACv2Env({"map_name": self.map_name, "seed": self.seed})


class PettingZooConfig(EnvConfig):
    def __init__(self, env_name, seed, continuous_action):
        self.env_name = env_name
        self.seed = seed
        self.continuous_action = continuous_action

    def create_env(self):
        from env.pettingzoo.mpe_env import PettingZooMPEEnv

        return PettingZooMPEEnv(self.env_name, self.seed, self.continuous_action)


class FootballConfig(EnvConfig):
    def __init__(self, env_name, seed):
        self.env_name = env_name
        self.seed = seed

    def create_env(self):
        from env.football.Football import Football

        return Football(self.env_name)


class MAMujocoConfig(EnvConfig):
    def __init__(self, scenario, seed, agent_conf, agent_obsk=0, episode_limit=1000):
        self.scenario = scenario
        self.seed = seed
        self.agent_conf = agent_conf
        self.agent_obsk = agent_obsk
        self.episode_limit = episode_limit

        self.env_args = {
            "random_seed": self.seed,
            "scenario": self.scenario,
            "agent_conf": self.agent_conf,
            "agent_obsk": self.agent_obsk,
            "episode_limit": self.episode_limit,
        }

    def create_env(self):
        from env.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

        return MujocoMulti(env_args=self.env_args)


class DexHandsConfig(EnvConfig):
    def __init__(
        self,
        task_name,
        seed,
        backend="isaacgym",
        num_envs=1,
        episode_limit=200,
        headless=True,
        rl_device="cuda:0",
        sim_device="cuda:0",
        pipeline="gpu",
    ):
        self.task_name = task_name
        self.seed = seed
        self.backend = backend
        self.num_envs = num_envs
        self.episode_limit = episode_limit
        self.headless = headless
        self.rl_device = rl_device
        self.sim_device = sim_device
        self.pipeline = pipeline

    def create_env(self):
        from env.bidexhands.bidexhands_env import DexHandsEnv

        return DexHandsEnv(
            task_name=self.task_name,
            seed=self.seed,
            backend=self.backend,
            num_envs=self.num_envs,
            episode_limit=self.episode_limit,
            headless=self.headless,
            rl_device=self.rl_device,
            sim_device=self.sim_device,
            pipeline=self.pipeline,
        )


class EnvCurriculumConfig(EnvConfig):
    def __init__(self, env_configs, env_episodes, env_type, obs_builder_config=None, reward_config=None):
        self.env_configs = env_configs
        self.env_episodes = env_episodes
        self.ENV_TYPE = env_type

        if obs_builder_config is not None:
            self.set_obs_builder_config(obs_builder_config)

        if reward_config is not None:
            self.set_reward_config(reward_config)

    def update_random_seed(self):
        for conf in self.env_configs:
            conf.update_random_seed()

    def set_obs_builder_config(self, obs_builder_config):
        for conf in self.env_configs:
            conf.set_obs_builder_config(obs_builder_config)

    def set_reward_config(self, reward_config):
        for conf in self.env_configs:
            conf.set_reward_config(reward_config)

    def create_env(self):
        return EnvCurriculum(self.env_configs, self.env_episodes)


class EnvCurriculumSampleConfig(EnvConfig):
    def __init__(self, env_configs, env_probs, obs_builder_config=None, reward_config=None):
        self.env_configs = env_configs
        self.env_probs = env_probs

        if obs_builder_config is not None:
            self.set_obs_builder_config(obs_builder_config)

        if reward_config is not None:
            self.set_reward_config(reward_config)

    def update_random_seed(self):
        for conf in self.env_configs:
            conf.update_random_seed()

    def set_obs_builder_config(self, obs_builder_config):
        for conf in self.env_configs:
            conf.set_obs_builder_config(obs_builder_config)

    def set_reward_config(self, reward_config):
        for conf in self.env_configs:
            conf.set_reward_config(reward_config)

    def create_env(self):
        return EnvCurriculumSample(self.env_configs, self.env_probs)


class EnvCurriculumPrioritizedSampleConfig(EnvConfig):
    def __init__(self, env_configs, repeat_random_seed, obs_builder_config=None, reward_config=None):
        self.env_configs = env_configs
        self.repeat_random_seed = repeat_random_seed

        if obs_builder_config is not None:
            self.set_obs_builder_config(obs_builder_config)

        if reward_config is not None:
            self.set_reward_config(reward_config)

    def update_random_seed(self):
        for conf in self.env_configs:
            conf.update_random_seed()

    def set_obs_builder_config(self, obs_builder_config):
        for conf in self.env_configs:
            conf.set_obs_builder_config(obs_builder_config)

    def set_reward_config(self, reward_config):
        for conf in self.env_configs:
            conf.set_reward_config(reward_config)

    def create_env(self):
        return EnvCurriculumPrioritizedSample(self.env_configs, self.repeat_random_seed)
