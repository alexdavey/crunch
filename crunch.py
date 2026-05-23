"""Helpers for loading scalar experiment data and pickled artifacts."""

import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import wandb
from tensorboard.backend.event_processing import event_accumulator


TensorboardRunData: TypeAlias = dict[str, Any]
TensorboardScalars: TypeAlias = dict[str, TensorboardRunData]
EventFileTask: TypeAlias = tuple[str, str, str | None]
WandbRunData: TypeAlias = dict[str, Any]


def find_event_files(log_dir: str) -> list[str]:
    """Return TensorBoard event files found recursively under ``log_dir``."""
    path = Path(log_dir).expanduser()
    event_files: list[str] = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))
    return event_files


def process_file(args: EventFileTask) -> TensorboardScalars:
    """Load scalar series from one TensorBoard event file.

    Args:
        args: Tuple containing the event file path, normalized log directory,
            and an optional scalar tag to retain.

    Returns:
        A single-run mapping keyed by the run's path relative to ``log_dir``.
        Each run includes ``full_filepath`` and one entry per retained scalar
        tag with ``steps`` and ``values`` arrays represented as lists.
    """
    filepath, log_dir, filter_tag = args
    run_name = os.path.relpath(os.path.dirname(filepath), log_dir)
    result: TensorboardScalars = {run_name: {"full_filepath": filepath}}
    try:
        ea = event_accumulator.EventAccumulator(filepath)
        ea.Reload()

        for tag in ea.Tags().get("scalars", []):
            if (filter_tag is not None) and (tag != filter_tag):
                continue
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            result[run_name][tag] = {"steps": steps, "values": values}
    except Exception as e:
        print(f"Error loading {filepath}: {e}")

    return result


def load_tensorboard_scalars(
    log_dir: str,
    filter_tag: str | None = None,
    include_empty: bool = False,
) -> TensorboardScalars:
    """Load scalar data from every TensorBoard event file under ``log_dir``.

    Args:
        log_dir: Directory containing TensorBoard event files, searched
            recursively.
        filter_tag: Optional scalar tag to keep. When omitted, all scalar tags
            are loaded.
        include_empty: Whether to include event files that contain no matching
            scalar tags.

    Raises:
        FileNotFoundError: If ``log_dir`` does not exist.
        NotADirectoryError: If ``log_dir`` is not a directory.

    Returns:
        Mapping from relative run names to their loaded scalar data.
    """
    path = Path(log_dir).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"log_dir does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"log_dir is not a directory: {path}")

    normalized_log_dir = str(path)
    event_files = find_event_files(normalized_log_dir)
    inputs = [(file, normalized_log_dir, filter_tag) for file in event_files]

    all_scalars_raw = Pool().map(process_file, inputs)

    all_scalars: TensorboardScalars = {}
    for d in all_scalars_raw:
        # Update if it's not just the filename
        for run_name, run in d.items():
            if len(run.keys()) != 1 or include_empty:
                assert run_name not in all_scalars
                all_scalars[run_name] = run

    return all_scalars


def load_wandb_scalars(
    tag: str,
    project: str,
    timeout: int = 30,
    **keys: str,
) -> list[WandbRunData]:
    """Load selected scalar histories for tagged Weights & Biases runs.

    Args:
        tag: W&B tag used to select runs.
        project: W&B project path, typically ``"entity/project"``.
        timeout: API timeout in seconds.
        **keys: Output field names mapped to W&B history keys. For example,
            ``accuracy="eval/accuracy"`` stores that history under
            ``"accuracy"`` in each returned run.

    Returns:
        A list of run dictionaries containing run metadata and NumPy arrays for
        each requested history key.
    """
    if not keys:
        raise ValueError("At least one key must be provided")

    api = wandb.Api(timeout=timeout)

    try:
        len(api.runs(path=project, per_page=1, include_sweeps=False))
    except ValueError as e:
        raise ValueError(
            f"W&B project does not exist or is not accessible: {project!r}"
        ) from e

    runs = api.runs(path=project, filters={"tags": tag})

    if len(runs) == 0:
        raise ValueError(f"No W&B runs found in {project!r} with tag {tag!r}")

    wandb_keys = list(keys.values())
    results: list[WandbRunData] = []

    for run in runs:
        collected = {name: [] for name in keys}

        data = run.scan_history(
            keys=wandb_keys,
            page_size=100000,
            min_step=None,
            max_step=None,
        )

        for entry in data:
            for out_name, wandb_key in keys.items():
                if wandb_key in entry:
                    collected[out_name].append(entry[wandb_key])

        result = {
            "name": run.name,
            "id": run.id,
            "seed": run.config.get("seed"),
        }

        for name, values in collected.items():
            result[name] = np.array(values)

        results.append(result)

    return results


def save_pickle(filename: str, data: Any) -> None:
    """Serialize ``data`` to ``filename`` using the highest pickle protocol."""
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename: str) -> Any:
    """Deserialize and return a Python object from ``filename``."""
    with open(filename, "rb") as f:
        return pickle.load(f)
