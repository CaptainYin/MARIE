import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_train_import_applies_thread_defaults_without_native_runtime():
    env = os.environ.copy()
    env["DIMA_DISABLE_NATIVE_RUNTIME"] = "1"
    for key in (
        "MARIE_TORCH_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env.pop(key, None)

    code = """
import os
import train
for key in ("MARIE_TORCH_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    print(f"{key}={os.environ.get(key)}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "MARIE_TORCH_NUM_THREADS=1" in result.stdout
    assert "OMP_NUM_THREADS=1" in result.stdout
    assert "MKL_NUM_THREADS=1" in result.stdout
    assert "OPENBLAS_NUM_THREADS=1" in result.stdout


def test_native_runtime_env_applies_thread_defaults():
    import native_runtime

    env, _ = native_runtime._build_native_runtime_env(
        env={},
        argv=["train.py"],
        prefix=Path(sys.prefix),
        py_version=f"python{sys.version_info.major}.{sys.version_info.minor}",
        home=Path.home(),
    )

    assert env["MARIE_TORCH_NUM_THREADS"] == "1"
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
