# Crunch

Utilities for loading experiment scalar data from Weights & Biases and
TensorBoard as pandas DataFrames, plus small pickle helpers.

## Installation

```bash
pip install "git+https://github.com/alexdavey/crunch.git"
```

## Weights & Biases

`load_wandb_scalars` loads selected history keys for all runs in a project with
a given tag. It returns a wide DataFrame by default: one row per run and W&B
`_step`, with one column per requested metric.

```python
from crunch import load_wandb_scalars

df = load_wandb_scalars(
    tag="baseline",
    project="entity/project",
    accuracy="eval/accuracy",
    loss="train/loss",
)
```

The DataFrame columns are `run_name`, `run_id`, `seed`, `step`, followed by the
requested metric names such as `accuracy` and `loss`. These metadata names are
reserved and cannot be used as requested metric names.

Pass `format="dict"` to return a list of dictionaries instead.

## TensorBoard

`load_tensorboard_scalars` loads scalar series from TensorBoard event files
under a log directory. It returns a wide DataFrame by default: one row per run
and step, with one column per TensorBoard scalar tag.

```python
from crunch import load_tensorboard_scalars

df = load_tensorboard_scalars("~/runs")
df = load_tensorboard_scalars("~/runs", filter_tag="val/accuracy")
```

The DataFrame columns are `run_name`, `full_filepath`, `step`, followed by the
scalar tag columns. These metadata names are reserved and cannot be used as
TensorBoard scalar tag names in DataFrame output.

Pass `format="dict"` to return a nested dictionary instead. In dict mode,
`include_empty=True` keeps event files without matching scalar data.

## Pickle Helpers

```python
from crunch import load_pickle, save_pickle

save_pickle("results.pkl", {"accuracy": 0.91})
results = load_pickle("results.pkl")
```
