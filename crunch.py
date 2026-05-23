"""Helpers for loading scalar experiment data and pickled artifacts."""

import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
import wandb
from tensorboard.backend.event_processing import event_accumulator


TensorboardRunData: TypeAlias = dict[str, Any]
TensorboardScalars: TypeAlias = dict[str, TensorboardRunData]
EventFileTask: TypeAlias = tuple[str, str, str | None]
WandbRunData: TypeAlias = dict[str, Any]
WANDB_METADATA_COLUMNS = {"run_name", "run_id", "seed", "step"}


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


def _wide_dataframe(
    rows: list[dict[str, Any]],
    index_columns: list[str],
    metric_column: str,
    value_column: str,
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Convert scalar rows to a wide-format DataFrame."""
    if metric_columns is None:
        metric_columns = sorted({row[metric_column] for row in rows})

    rows_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[column] for column in index_columns)
        wide_row = rows_by_key.setdefault(
            key,
            {column: row[column] for column in index_columns},
        )
        wide_row[row[metric_column]] = row[value_column]

    return pd.DataFrame(
        sorted(
            rows_by_key.values(),
            key=lambda row: tuple(
                (row[column] is None, row[column]) for column in index_columns
            ),
        ),
        columns=[*index_columns, *metric_columns],
    )


def _tensorboard_scalars_to_dataframe(scalars: TensorboardScalars) -> pd.DataFrame:
    """Convert TensorBoard scalar data to a wide-format DataFrame."""
    metadata_columns = ["run_name", "full_filepath", "step"]
    rows: list[dict[str, Any]] = []
    for run_name, run in scalars.items():
        full_filepath = run["full_filepath"]
        for tag, series in run.items():
            if tag == "full_filepath":
                continue
            if tag in metadata_columns:
                raise ValueError(
                    f"TensorBoard scalar tag conflicts with DataFrame metadata "
                    f"column: {tag!r}"
                )
            for step, value in zip(series["steps"], series["values"]):
                rows.append(
                    {
                        "run_name": run_name,
                        "full_filepath": full_filepath,
                        "step": step,
                        "tag": tag,
                        "value": value,
                    }
                )

    return _wide_dataframe(rows, metadata_columns, "tag", "value")


def load_tensorboard_scalars(
    log_dir: str,
    filter_tag: str | None = None,
    include_empty: bool = False,
    *,
    format: str = "dataframe",
) -> TensorboardScalars | pd.DataFrame:
    """Load scalar data from every TensorBoard event file under ``log_dir``.

    Args:
        log_dir: Directory containing TensorBoard event files, searched
            recursively.
        filter_tag: Optional scalar tag to keep. When omitted, all scalar tags
            are loaded.
        include_empty: Whether to include event files that contain no matching
            scalar tags in ``format="dict"`` output. Empty runs produce no rows
            in ``format="dataframe"`` output.
        format: Return format. Use ``"dataframe"`` for a wide-format pandas
            DataFrame or ``"dict"`` for the nested mapping.

    Returns:
        A DataFrame with ``run_name``, ``full_filepath``, ``step``, and one
        column per scalar tag, or a mapping from relative run names to loaded
        scalar data when ``format="dict"``.
    """
    if format not in {"dict", "dataframe"}:
        raise ValueError("format must be 'dict' or 'dataframe'")

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

    if format == "dataframe":
        return _tensorboard_scalars_to_dataframe(all_scalars)

    return all_scalars


def load_wandb_scalars(
    tag: str,
    project: str,
    timeout: int = 30,
    *,
    format: str = "dataframe",
    **keys: str,
) -> list[WandbRunData] | pd.DataFrame:
    """Load selected scalar histories for tagged Weights & Biases runs.

    Args:
        tag: W&B tag used to select runs.
        project: W&B project path, typically ``"entity/project"``.
        timeout: API timeout in seconds.
        format: Return format. Use ``"dataframe"`` for a wide-format pandas
            DataFrame or ``"dict"`` for a list of run dictionaries.
        **keys: Output field names mapped to W&B history keys. For example,
            ``accuracy="eval/accuracy"`` stores that history under
            ``"accuracy"`` in each returned run.

    Returns:
        A DataFrame with ``run_name``, ``run_id``, ``seed``, ``step``, and one
        column per requested output key, or a list of run dictionaries when
        ``format="dict"``.
    """
    if not keys:
        raise ValueError("At least one key must be provided")
    if format not in {"dict", "dataframe"}:
        raise ValueError("format must be 'dict' or 'dataframe'")
    if format == "dataframe":
        conflicting_keys = set(keys) & WANDB_METADATA_COLUMNS
        if conflicting_keys:
            conflicts = ", ".join(sorted(repr(key) for key in conflicting_keys))
            raise ValueError(
                f"W&B output keys conflict with DataFrame metadata columns: {conflicts}"
            )

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
    history_keys = wandb_keys
    if format == "dataframe":
        history_keys = list(dict.fromkeys([*wandb_keys, "_step"]))
    results: list[WandbRunData] = []
    dataframe_rows: list[dict[str, Any]] = []

    for run in runs:
        collected = {name: [] for name in keys}

        data = run.scan_history(
            keys=history_keys,
            page_size=100000,
            min_step=None,
            max_step=None,
        )

        for entry in data:
            for out_name, wandb_key in keys.items():
                if wandb_key in entry:
                    collected[out_name].append(entry[wandb_key])
                    if format == "dataframe":
                        dataframe_rows.append(
                            {
                                "run_name": run.name,
                                "run_id": run.id,
                                "seed": run.config.get("seed"),
                                "step": entry.get("_step"),
                                "metric": out_name,
                                "value": entry[wandb_key],
                            }
                        )

        result = {
            "name": run.name,
            "id": run.id,
            "seed": run.config.get("seed"),
        }

        for name, values in collected.items():
            result[name] = np.array(values)

        results.append(result)

    if format == "dataframe":
        return _wide_dataframe(
            dataframe_rows,
            ["run_name", "run_id", "seed", "step"],
            "metric",
            "value",
            list(keys),
        )

    return results


def save_pickle(filename: str, data: Any) -> None:
    """Serialize ``data`` to ``filename`` using the highest pickle protocol."""
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename: str) -> Any:
    """Deserialize and return a Python object from ``filename``."""
    with open(filename, "rb") as f:
        return pickle.load(f)
