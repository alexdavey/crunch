import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from tensorboard.backend.event_processing import event_accumulator


# Recursively find all event files in a directory
def find_event_files(log_dir: str):
    path = Path(log_dir).expanduser()
    event_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))
    return event_files


def process_file(args: dict):
    filepath, log_dir, filter_tag = args
    run_name = os.path.relpath(os.path.dirname(filepath), log_dir)
    result = {run_name: {"full_filepath": filepath}}
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


# Load scalar data from each event file
def load_tensorboard_scalars(
    log_dir: str,
    filter_tag: str | None = None,
    include_empty: bool = False,
):
    path = Path(log_dir).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"log_dir does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"log_dir is not a directory: {path}")

    normalized_log_dir = str(path)
    event_files = find_event_files(normalized_log_dir)
    inputs = [(file, normalized_log_dir, filter_tag) for file in event_files]

    all_scalars_raw = Pool().map(process_file, inputs)

    all_scalars = {}
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
):
    if not keys:
        raise ValueError("At least one key must be provided")

    api = wandb.Api(timeout=timeout)
    runs = api.runs(path=project, filters={"tags": tag})

    wandb_keys = list(keys.values())
    results = []

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


def save_pickle(filename: str, data: Any):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)
