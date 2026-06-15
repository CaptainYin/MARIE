"""Thread runtime defaults for MARIE training entrypoints."""

import os
from typing import MutableMapping


TORCH_THREAD_ENV = "MARIE_TORCH_NUM_THREADS"
DEFAULT_THREAD_LIMIT = "1"
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def apply_thread_env_defaults(env: MutableMapping[str, str] = os.environ) -> MutableMapping[str, str]:
    env.setdefault(TORCH_THREAD_ENV, DEFAULT_THREAD_LIMIT)
    for thread_env in THREAD_ENV_VARS:
        env.setdefault(thread_env, env[TORCH_THREAD_ENV])
    return env


def get_torch_thread_count(env: MutableMapping[str, str] = os.environ) -> int:
    thread_count = int(env.get(TORCH_THREAD_ENV, DEFAULT_THREAD_LIMIT))
    if thread_count < 1:
        raise ValueError(f"{TORCH_THREAD_ENV} must be >= 1")
    return thread_count


def configure_torch_runtime(torch_module, env: MutableMapping[str, str] = os.environ) -> None:
    apply_thread_env_defaults(env)
    thread_count = get_torch_thread_count(env)
    torch_module.set_num_threads(thread_count)
    try:
        torch_module.set_num_interop_threads(thread_count)
    except RuntimeError:
        pass
