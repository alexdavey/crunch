# Crunch

Small utilities for collecting experiment scalar data from TensorBoard and
Weights & Biases, plus simple pickle save/load helpers.

## Installation

Install the package in editable mode from this directory:

```bash
pip install -e .
```

The module imports `numpy`, `wandb`, and TensorBoard's event accumulator, so make
sure those packages are available in the Python environment where you use it.

## Weights & Biases Scalars

Load selected history keys for all runs in a project that have a given tag:

```python
from crunch import load_wandb_scalars

runs = load_wandb_scalars(
    tag="baseline",
    project="entity/project",
    accuracy="eval/accuracy",
    loss="train/loss",
)

for run in runs:
    print(run["name"], run["id"], run["seed"])
    print(run["accuracy"])
```

Keyword argument names become output field names. Keyword argument values are
the W&B history keys to scan.

`load_wandb_scalars` returns a list of dictionaries, one dictionary per W&B run.
Each dictionary contains:

- `name`: the W&B run name.
- `id`: the W&B run id.
- `seed`: the `seed` value from the run config, or `None` if it is absent.
- One NumPy array per requested scalar, keyed by the keyword argument name used
  in the call.

For example, this call:

```python
runs = load_wandb_scalars(
    tag="baseline",
    project="entity/project",
    accuracy="eval/accuracy",
)
```

returns data shaped like:

```python
[
    {
        "name": "run-1",
        "id": "abc123",
        "seed": 0,
        "accuracy": np.array([...]),
    },
]
```

## TensorBoard Scalars

Load all scalar series from TensorBoard event files under a log directory:

```python
from crunch import load_tensorboard_scalars

scalars = load_tensorboard_scalars("~/runs")

for run_name, run_data in scalars.items():
    print(run_name, run_data["full_filepath"])
    for tag, series in run_data.items():
        if tag == "full_filepath":
            continue
        print(tag, series["steps"], series["values"])
```

Restrict loading to one scalar tag:

```python
scalars = load_tensorboard_scalars("~/runs", filter_tag="val/accuracy")
```

By default, event files without matching scalar data are skipped. Set
`include_empty=True` to keep them in the returned mapping.

`load_tensorboard_scalars` returns a dictionary keyed by run name. Each run name
is the event file's parent directory relative to the input `log_dir`. Each value
is another dictionary containing:

- `full_filepath`: the path to the TensorBoard event file.
- One entry per scalar tag. Each tag maps to a dictionary with `steps` and
  `values` lists.

For example:

```python
{
    "experiment-1/seed-0": {
        "full_filepath": "/logs/experiment-1/seed-0/events.out.tfevents...",
        "val/accuracy": {
            "steps": [0, 1, 2],
            "values": [0.42, 0.55, 0.61],
        },
    },
}
```

## Pickle Helpers

```python
from crunch import load_pickle, save_pickle

save_pickle("results.pkl", {"accuracy": 0.91})
results = load_pickle("results.pkl")
```

`save_pickle` uses the highest pickle protocol available in the current Python
runtime.
