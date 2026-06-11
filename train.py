import argparse
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for matplotlib
import datetime
import os
import random
import shutil
from pathlib import Path
import native_runtime
native_runtime.prepare_train_process()
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np

from environments import Env


def configure_wandb_environment() -> None:
    os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
    os.environ.setdefault("WANDB_START_METHOD", "thread")


configure_wandb_environment()

DreamerRunner = None
Experiment = None
EnvCurriculumConfig = None
StarCraftConfig = None
SMAXConfig = None
SMACv2Config = None
PettingZooConfig = None
FootballConfig = None
MAMujocoConfig = None
DexHandsConfig = None
DreamerControllerConfig = None
DreamerLearnerConfig = None
MPEDreamerLearnerConfig = None
MPEDreamerControllerConfig = None
GRFDreamerLearnerConfig = None
GRFDreamerControllerConfig = None
MAMujocoDreamerLearnerConfig = None
MAMujocoDreamerControllerConfig = None
SMAXDreamerLearnerConfig = None
SMAXDreamerControllerConfig = None
generate_group_name = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="flatland", help="Flatland or SMAC env")
    parser.add_argument("--env_name", type=str, default="5_agents", help="Specific setting")

    parser.add_argument("--agent_conf", type=str, default=None)
    parser.add_argument("--enable_mpe_disc", action="store_true")

    parser.add_argument("--n_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--seed", type=int, default=1, help="Random seed id")
    parser.add_argument("--steps", type=int, default=1e6, help="Max environment steps")
    parser.add_argument("--mode", type=str, default="disabled")
    parser.add_argument("--tokenizer", type=str, default="vq")
    parser.add_argument("--decay", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--sample_temp", type=float, default="inf")
    parser.add_argument("--model_epochs", type=int, default=None)
    parser.add_argument("--wm_epochs", type=int, default=None)
    parser.add_argument("--agent_epochs", type=int, default=None)

    parser.add_argument("--average_r", action="store_true")
    parser.add_argument("--ce_for_r", action="store_true")
    parser.add_argument("--ce_for_av", action="store_true")
    parser.add_argument("--ce_for_end", action="store_true")
    parser.add_argument(
        "--rl_device",
        type=str,
        default="cuda:0",
        help="RL device for bidexhands, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--sim_device",
        type=str,
        default="cuda:0",
        help="IsaacGym sim device for bidexhands, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="gpu",
        choices=("gpu", "cpu", "cuda"),
        help="IsaacGym pipeline mode for bidexhands",
    )

    return parser.parse_args()


def _seed_everywhere(seed: int, torch_module) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed(seed)
        torch_module.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False


def _preimport_bidexhands_isaacgym() -> None:
    from env.bidexhands.bidexhands_env import DexHandsEnv

    DexHandsEnv._patch_numpy_deprecated_aliases()
    DexHandsEnv._import_bidexhands_with_isaacgym_guard()


def _lazy_import_training_modules() -> None:
    global DreamerRunner, Experiment, EnvCurriculumConfig
    global StarCraftConfig, SMAXConfig, SMACv2Config, PettingZooConfig, FootballConfig, MAMujocoConfig, DexHandsConfig
    global DreamerControllerConfig, DreamerLearnerConfig
    global MPEDreamerLearnerConfig, MPEDreamerControllerConfig
    global GRFDreamerLearnerConfig, GRFDreamerControllerConfig
    global MAMujocoDreamerLearnerConfig, MAMujocoDreamerControllerConfig
    global SMAXDreamerLearnerConfig, SMAXDreamerControllerConfig
    global generate_group_name

    if DreamerRunner is not None:
        return

    from agent.runners.DreamerRunner import DreamerRunner as _DreamerRunner
    from configs import Experiment as _Experiment
    from configs.EnvConfigs import (
        DexHandsConfig as _DexHandsConfig,
        EnvCurriculumConfig as _EnvCurriculumConfig,
        FootballConfig as _FootballConfig,
        MAMujocoConfig as _MAMujocoConfig,
        PettingZooConfig as _PettingZooConfig,
        SMACv2Config as _SMACv2Config,
        SMAXConfig as _SMAXConfig,
        StarCraftConfig as _StarCraftConfig,
    )
    from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig as _DreamerControllerConfig
    from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig as _DreamerLearnerConfig
    from configs.dreamer.football.GRFControllerConfig import GRFDreamerControllerConfig as _GRFDreamerControllerConfig
    from configs.dreamer.football.GRFLearnerConfig import GRFDreamerLearnerConfig as _GRFDreamerLearnerConfig
    from configs.dreamer.mamujoco.mamujocoControllerConfig import (
        MAMujocoDreamerControllerConfig as _MAMujocoDreamerControllerConfig,
    )
    from configs.dreamer.mamujoco.mamujocoLearnerConfig import (
        MAMujocoDreamerLearnerConfig as _MAMujocoDreamerLearnerConfig,
    )
    from configs.dreamer.mpe.MpeControllerConfig import MPEDreamerControllerConfig as _MPEDreamerControllerConfig
    from configs.dreamer.mpe.MpeLearnerConfig import MPEDreamerLearnerConfig as _MPEDreamerLearnerConfig
    from configs.dreamer.smax.SMAXControllerConfig import SMAXDreamerControllerConfig as _SMAXDreamerControllerConfig
    from configs.dreamer.smax.SMAXLearnerConfig import SMAXDreamerLearnerConfig as _SMAXDreamerLearnerConfig
    from utils import generate_group_name as _generate_group_name

    DreamerRunner = _DreamerRunner
    Experiment = _Experiment
    EnvCurriculumConfig = _EnvCurriculumConfig
    StarCraftConfig = _StarCraftConfig
    SMAXConfig = _SMAXConfig
    SMACv2Config = _SMACv2Config
    PettingZooConfig = _PettingZooConfig
    FootballConfig = _FootballConfig
    MAMujocoConfig = _MAMujocoConfig
    DexHandsConfig = _DexHandsConfig
    DreamerControllerConfig = _DreamerControllerConfig
    DreamerLearnerConfig = _DreamerLearnerConfig
    MPEDreamerLearnerConfig = _MPEDreamerLearnerConfig
    MPEDreamerControllerConfig = _MPEDreamerControllerConfig
    GRFDreamerLearnerConfig = _GRFDreamerLearnerConfig
    GRFDreamerControllerConfig = _GRFDreamerControllerConfig
    MAMujocoDreamerLearnerConfig = _MAMujocoDreamerLearnerConfig
    MAMujocoDreamerControllerConfig = _MAMujocoDreamerControllerConfig
    SMAXDreamerLearnerConfig = _SMAXDreamerLearnerConfig
    SMAXDreamerControllerConfig = _SMAXDreamerControllerConfig
    generate_group_name = _generate_group_name


def train_dreamer(exp, n_workers):
    runner = DreamerRunner(exp.env_config, exp.learner_config, exp.controller_config, n_workers)
    runner.run(exp.steps, exp.episodes, save_interval=20000, save_mode="interval")


def _shutdown_runtime(wandb_run) -> None:
    # Tear down Ray before closing wandb so background worker output and thread
    # cleanup do not race with wandb's async upload pool during interpreter exit.
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

    if wandb_run is not None:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass


def get_env_info(configs, env):
    if not env.discrete:
        assert hasattr(env, "individual_action_space")
        individual_action_space = env.individual_action_space
    else:
        individual_action_space = None

    for config in configs:
        config.IN_DIM = env.n_obs
        config.ACTION_SIZE = env.n_actions
        config.NUM_AGENTS = env.n_agents
        config.CONTINUOUS_ACTION = not env.discrete
        config.ACTION_SPACE = individual_action_space

    print(f"Observation dims: {env.n_obs}")
    print(f"Action dims: {env.n_actions}")
    print(f"Num agents: {env.n_agents}")
    print(f"Continuous action for control? -> {not env.discrete}")

    if hasattr(env, "individual_action_space"):
        print(f"Individual action space: {env.individual_action_space}")

    env.close()


def prepare_starcraft_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = StarCraftConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {
        "env_config": (env_config, 2000),
        "controller_config": agent_configs[0],
        "learner_config": agent_configs[1],
        "reward_config": None,
        "obs_builder_config": None,
    }


def prepare_pettingzoo_configs(env_name, continuous_action=True):
    agent_configs = [MPEDreamerControllerConfig(), MPEDreamerLearnerConfig()]
    env_config = PettingZooConfig(env_name, RANDOM_SEED, continuous_action)
    get_env_info(agent_configs, env_config.create_env())
    return {
        "env_config": (env_config, 5000),
        "controller_config": agent_configs[0],
        "learner_config": agent_configs[1],
        "reward_config": None,
        "obs_builder_config": None,
    }


def prepare_football_configs(env_name):
    agent_configs = [GRFDreamerControllerConfig(), GRFDreamerLearnerConfig()]
    env_config = FootballConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {
        "env_config": (env_config, 5000),
        "controller_config": agent_configs[0],
        "learner_config": agent_configs[1],
        "reward_config": None,
        "obs_builder_config": None,
    }


def prepare_mamujoco_configs(scenario, agent_config):
    agent_configs = [MAMujocoDreamerControllerConfig(), MAMujocoDreamerLearnerConfig()]
    env_config = MAMujocoConfig(scenario=scenario, seed=RANDOM_SEED, agent_conf=agent_config)
    get_env_info(agent_configs, env_config.create_env())
    return {
        "env_config": (env_config, 5000),
        "controller_config": agent_configs[0],
        "learner_config": agent_configs[1],
        "reward_config": None,
        "obs_builder_config": None,
    }


def prepare_smax_configs(env_name):
    agent_configs = [SMAXDreamerControllerConfig(), SMAXDreamerLearnerConfig()]
    env_config = SMAXConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {
        "env_config": (env_config, 5000),
        "controller_config": agent_configs[0],
        "learner_config": agent_configs[1],
        "reward_config": None,
        "obs_builder_config": None,
    }


def prepare_smacv2_configs(env_name):
    agent_configs = [SMAXDreamerControllerConfig(), SMAXDreamerLearnerConfig()]
    env_config = SMACv2Config(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {
        "env_config": (env_config, 2000),
        "controller_config": agent_configs[0],
        "learner_config": agent_configs[1],
        "reward_config": None,
        "obs_builder_config": None,
    }


def prepare_bidexhands_configs(task_name, rl_device="cuda:0", sim_device="cuda:0", pipeline="gpu"):
    from gym.spaces import Box

    agent_configs = [MAMujocoDreamerControllerConfig(), MAMujocoDreamerLearnerConfig()]
    env_config = DexHandsConfig(
        task_name=task_name,
        seed=RANDOM_SEED,
        rl_device=rl_device,
        sim_device=sim_device,
        pipeline=pipeline,
    )

    if task_name == "ShadowHandBottleCap":
        in_dim, action_size = 221, 26
    elif task_name in ("ShadowHandDoorOpenInward", "ShadowHandDoorOpenOutward", "ShadowHandPen"):
        in_dim, action_size = 218, 26
    else:
        raise ValueError(
            f"Unsupported bidexhands task '{task_name}'. "
            "Supported tasks: ShadowHandBottleCap, ShadowHandDoorOpenInward, "
            "ShadowHandDoorOpenOutward, ShadowHandPen"
        )

    action_space = Box(low=-1.0, high=1.0, shape=(action_size,), dtype=np.float32)
    for config in agent_configs:
        config.IN_DIM = in_dim
        config.ACTION_SIZE = action_size
        config.NUM_AGENTS = 2
        config.CONTINUOUS_ACTION = True
        config.ACTION_SPACE = action_space
        if hasattr(config, "DEVICE"):
            config.DEVICE = rl_device
        if hasattr(config, "MODEL_BATCH_SIZE"):
            config.MODEL_BATCH_SIZE = min(config.MODEL_BATCH_SIZE, 8)
        if hasattr(config, "BATCH_SIZE"):
            config.BATCH_SIZE = min(config.BATCH_SIZE, 8)
        if hasattr(config, "ac_batch_size"):
            config.ac_batch_size = min(config.ac_batch_size, 128)

    return {
        "env_config": (env_config, 5000),
        "controller_config": agent_configs[0],
        "learner_config": agent_configs[1],
        "reward_config": None,
        "obs_builder_config": None,
    }


if __name__ == "__main__":
    RANDOM_SEED = 23
    args = parse_args()

    if args.env == "dexhands":
        args.env = Env.BIDEXHANDS.value

    if args.env == Env.BIDEXHANDS.value:
        _preimport_bidexhands_isaacgym()

    _lazy_import_training_modules()

    import torch

    RANDOM_SEED += args.seed * 100
    if args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.env_name)
    elif args.env == Env.SMAX:
        configs = prepare_smax_configs(args.env_name)
    elif args.env == Env.SMACv2:
        configs = prepare_smacv2_configs(args.env_name)
    elif args.env == Env.PETTINGZOO:
        configs = prepare_pettingzoo_configs(
            args.env_name, continuous_action=not args.enable_mpe_disc
        )
    elif args.env == Env.GRF:
        configs = prepare_football_configs(args.env_name)
    elif args.env == Env.MAMUJOCO:
        configs = prepare_mamujoco_configs(args.env_name, args.agent_conf)
    elif args.env == Env.BIDEXHANDS:
        configs = prepare_bidexhands_configs(
            args.env_name,
            rl_device=args.rl_device,
            sim_device=args.sim_device,
            pipeline=args.pipeline,
        )
    else:
        raise Exception("Unknown environment")

    _seed_everywhere(RANDOM_SEED, torch)
    torch.autograd.set_detect_anomaly(False)

    configs["env_config"][0].ENV_TYPE = Env(args.env)
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)

    configs["learner_config"].seed = RANDOM_SEED

    if args.model_epochs is not None:
        configs["learner_config"].MODEL_EPOCHS = args.model_epochs
    if args.wm_epochs is not None:
        configs["learner_config"].WM_EPOCHS = args.wm_epochs
    if args.agent_epochs is not None:
        configs["learner_config"].EPOCHS = args.agent_epochs

    configs["learner_config"].tokenizer_type = args.tokenizer
    configs["controller_config"].tokenizer_type = args.tokenizer
    configs["learner_config"].ema_decay = args.decay
    configs["controller_config"].ema_decay = args.decay

    configs["controller_config"].temperature = args.temperature

    configs["learner_config"].critic_average_r = args.average_r

    configs["learner_config"].use_ce_for_r = args.ce_for_r
    configs["learner_config"].use_ce_for_end = False
    configs["learner_config"].use_ce_for_av_action = args.ce_for_av

    if args.sample_temp == float("inf"):
        configs["learner_config"].sample_temperature = str(args.sample_temp)
    else:
        configs["learner_config"].sample_temperature = args.sample_temp

    current_date = datetime.datetime.now()
    current_date_string = current_date.strftime("%m%d")

    dir_prefix = args.env_name + "-" + args.agent_conf if args.agent_conf is not None else args.env_name
    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + f"/{current_date_string}_results") / args.env / (dir_prefix + f"-{args.tokenizer}")
    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        curr_run = "run1" if len(exst_run_nums) == 0 else f"run{max(exst_run_nums) + 1}"

    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
        os.makedirs(str(run_dir / "ckpt"))

    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "agent"), dst=run_dir / "agent")
    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "configs"), dst=run_dir / "configs")
    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "networks"), dst=run_dir / "networks")
    shutil.copyfile(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "train.py"), dst=run_dir / "train.py")

    print(f"Run files are saved at {str(run_dir)}\n")

    configs["learner_config"].RUN_DIR = str(run_dir)
    configs["learner_config"].map_name = args.env_name

    group_name = generate_group_name(args, configs["learner_config"])

    if args.env == Env.MAMUJOCO:
        run_name = f"MARIE_s{args.seed}_{args.env_name}"
        if args.agent_conf is not None:
            run_name += f"_{args.agent_conf}"
    else:
        run_name = f"MARIE_s{args.seed}_{args.env_name}"

    job_type = "MARIE"

    import wandb

    if args.env == Env.MAMUJOCO:
        project_name = "mamujoco"
    elif args.env == Env.PETTINGZOO:
        project_name = "MPE"
    elif args.env == Env.SMACv2:
        project_name = "SMACv2"
    elif args.env == Env.BIDEXHANDS:
        project_name = "dexhands"
    else:
        project_name = "SMAD"

    wandb_run = wandb.init(
        project=project_name,
        mode=args.mode,
        group=group_name,
        job_type=job_type,
        name=run_name,
        config=configs["learner_config"].to_dict(),
        notes="",
    )

    try:
        exp = Experiment(
            steps=args.steps,
            episodes=50000,
            random_seed=RANDOM_SEED,
            env_config=EnvCurriculumConfig(
                *zip(configs["env_config"]),
                Env(args.env),
                obs_builder_config=configs["obs_builder_config"],
                reward_config=configs["reward_config"],
            ),
            controller_config=configs["controller_config"],
            learner_config=configs["learner_config"],
        )

        train_dreamer(exp, n_workers=args.n_workers)
    finally:
        _shutdown_runtime(wandb_run)
