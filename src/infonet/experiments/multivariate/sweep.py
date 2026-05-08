"""Parallel sweep of fully-conditional TE estimators across (n, ord_bins, rep)."""
from __future__ import annotations

import gc
import hashlib
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from infonet.core import Ordinal
from infonet.multivariate import (
    MultivariatePipeline,
    VAR1, NVAR,
    FullCondTE_Gaussian_SigTest,
    FullCondTE_KSG_PermTest,
    mv_results_to_df,
)

from infonet.experiments.multivariate.combine import combine


OUTDIR = Path("./results/multivariate")
CHECKPOINT_DIR = OUTDIR / "checkpoints"
COMBINED_PATH = OUTDIR / "combined.parquet"

N_LEVELS = [50, 100, 200, 500, 1000]
ORD_LEVELS = [None, 20, 10, 5, 3, 2]
N_PERM = 500  # for permutation test, lower value (say, 100) is recommended
D = 5
REPS = 100
SEED = 42
CORRECTION_ALPHA = 0.10

_GROUP_ENV = "SWEEP_ESTIMATOR_GROUP"

ESTIMATOR_GROUPS = {
    "gaussian": [
        ("FullCondTE_Gaussian_Sig",
            lambda: FullCondTE_Gaussian_SigTest(alpha=CORRECTION_ALPHA, correction="fdr_bh")),
    ],
    "ksg": [
        ("FullCondTE_KSG_Perm",
            lambda: FullCondTE_KSG_PermTest(k=4, n_perm=500, alpha=CORRECTION_ALPHA, correction="fdr_bh")),
    ],
    "all": [
        ("FullCondTE_Gaussian_Sig",
            lambda: FullCondTE_Gaussian_SigTest(alpha=CORRECTION_ALPHA, correction="fdr_bh")),
        ("FullCondTE_KSG_Perm",
            lambda: FullCondTE_KSG_PermTest(k=4, n_perm=500, alpha=CORRECTION_ALPHA, correction="fdr_bh")),
    ],
}

_active_group = os.environ.get(_GROUP_ENV, "all")
if _active_group not in ESTIMATOR_GROUPS:
    raise ValueError(
        f"Unknown estimator group from env {_GROUP_ENV}={_active_group!r}. "
        f"Available: {list(ESTIMATOR_GROUPS)}"
    )
ESTIMATOR_FACTORIES = ESTIMATOR_GROUPS[_active_group]

print(f"[PID {os.getpid()}] Active estimator group: {_active_group!r} "
      f"({[n for n, _ in ESTIMATOR_FACTORIES]})")

GENERATOR_FACTORIES = [
    ("VAR1", lambda: VAR1(sigma=1.0)),
    ("NVAR_r2.8", lambda: NVAR(r=2.8, sigma=0.1, boundary="reflect", max_radius=0.95, A_upper_scale=0.5)),
    ("NVAR_r3.2", lambda: NVAR(r=3.2, sigma=0.1, boundary="reflect", max_radius=0.95, A_upper_scale=0.5)),
    ("NVAR_r3.6", lambda: NVAR(r=3.6, sigma=0.1, boundary="reflect", max_radius=0.95, A_upper_scale=0.5)),
]


def _task_seed(est: str, gen: str, n: int, ord_bins, rep: int) -> int:
    ord_tag = "cont" if ord_bins is None else f"bins{ord_bins}"
    key = f"{SEED}|{est}|{gen}|n{n}|{ord_tag}|{rep}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def _checkpoint_path(est: str, gen: str, n: int, ord_bins, rep: int) -> Path:
    ord_tag = "cont" if ord_bins is None else f"bins{ord_bins}"
    return CHECKPOINT_DIR / f"est={est}__gen={gen}__n={n}__ord={ord_tag}__rep={rep}.parquet"


def _lookup_estimator_factory(name: str):
    for n, factory in ESTIMATOR_FACTORIES:
        if n == name:
            return factory
    raise KeyError(f"Unknown estimator factory: {name}")


def _lookup_generator_factory(name: str):
    for n, factory in GENERATOR_FACTORIES:
        if n == name:
            return factory
    raise KeyError(f"Unknown generator factory: {name}")


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


@contextmanager
def suppress_fd_all():
    """Silence both Python-level sys.stdout/stderr AND OS-level FDs."""
    fd_out = fd_err = devnull = None
    try:
        try:
            fd_out = os.dup(1)
            fd_err = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
        except OSError:
            devnull = None

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield
    finally:
        if fd_out is not None:
            os.dup2(fd_out, 1)
            os.close(fd_out)
        if fd_err is not None:
            os.dup2(fd_err, 2)
            os.close(fd_err)
        if devnull is not None:
            os.close(devnull)


def run_one(est_name, gen_name, n, ord_bins, rep):
    seed = _task_seed(est_name, gen_name, n, ord_bins, rep)

    pipe = (
        MultivariatePipeline()
        .generators(_lookup_generator_factory(gen_name)())
        .estimators(_lookup_estimator_factory(est_name)())
        .verbose(False)
    )
    if ord_bins is not None:
        pipe = pipe.sensitivities(Ordinal(n_bins=ord_bins))

    results = pipe.run(n=n, d=D, seed=seed, reps=1)
    df = mv_results_to_df(results)
    df["rep"] = rep
    return df


def _worker(task):
    est_name, gen_name, n, ord_bins, rep = task
    t0 = time.perf_counter()
    try:
        with suppress_fd_all():
            df = run_one(est_name, gen_name, n, ord_bins, rep)
        _atomic_write_parquet(df, _checkpoint_path(est_name, gen_name, n, ord_bins, rep))
        return (task, time.perf_counter() - t0, None)
    except Exception as e:
        return (task, time.perf_counter() - t0, f"{type(e).__name__}: {e}")
    finally:
        gc.collect()


def _default_n_workers() -> int:
    for env_var in ("SWEEP_WORKERS", "SLURM_CPUS_PER_TASK"):
        val = os.environ.get(env_var)
        if val:
            try:
                return max(1, int(val))
            except ValueError:
                pass
    cpus = os.cpu_count()
    return max(1, cpus - 1) if cpus else 1


def run_all(
    total_reps: int = None,
    n_levels: list[int] = None,
    ord_levels: list = None,
    n_workers: int = None,
    estimator_group: str | None = None,
    show_skipped: bool = True,
) -> None:
    global ESTIMATOR_FACTORIES
    if estimator_group is not None:
        if estimator_group not in ESTIMATOR_GROUPS:
            raise ValueError(
                f"Unknown estimator group {estimator_group!r}. "
                f"Available: {list(ESTIMATOR_GROUPS)}"
            )
        os.environ[_GROUP_ENV] = estimator_group
        ESTIMATOR_FACTORIES = ESTIMATOR_GROUPS[estimator_group]

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if total_reps is None:
        total_reps = REPS
    if n_workers is None:
        n_workers = _default_n_workers()
    if n_levels is None:
        n_levels = N_LEVELS
    if ord_levels is None:
        ord_levels = ORD_LEVELS

    tasks = [
        (est_name, gen_name, n, ord_bins, rep)
        for est_name, _ in ESTIMATOR_FACTORIES
        for gen_name, _ in GENERATOR_FACTORIES
        for n in n_levels
        for ord_bins in ord_levels
        for rep in range(total_reps)
    ]

    todo, skipped = [], 0
    for t in tasks:
        if _checkpoint_path(*t).exists():
            skipped += 1
        else:
            todo.append(t)

    if show_skipped and skipped:
        tqdm.write(f"Resuming: {skipped} / {len(tasks)} units already complete.")

    if not todo:
        tqdm.write("Nothing to do.")
        return

    tqdm.write(f"Running {len(todo)} units across {n_workers} workers.")

    with ProcessPoolExecutor(max_workers=n_workers) as executor, \
         tqdm(total=len(todo), desc="units", unit="unit",
              dynamic_ncols=True, leave=True) as pbar:

        futures = {executor.submit(_worker, t): t for t in todo}
        for future in as_completed(futures):
            task, elapsed, error = future.result()
            est_name, gen_name, n, ord_bins, rep = task
            ord_tag = "cont" if ord_bins is None else f"bins{ord_bins}"

            if error:
                tqdm.write(
                    f"[error] {est_name} / {gen_name} / n={n} / "
                    f"{ord_tag} / rep={rep}: {error}"
                )
            else:
                pbar.set_postfix_str(
                    f"{est_name}/{gen_name}/n={n}/{ord_tag}/rep={rep}  ({elapsed:.1f}s)",
                    refresh=False,
                )
            pbar.update(1)


if __name__ == "__main__":
    run_all(estimator_group="gaussian", n_workers=8)
    run_all(estimator_group="ksg", n_workers=3)
    df = combine()
    tqdm.write(f"Done. {len(df)} rows.")