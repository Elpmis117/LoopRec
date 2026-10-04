# LoopRec

Implementation of **LoopRec**, proposed in
**Think Harder Where It Matters: Item-Level Adaptive Recursion for Sequential Recommendation**.
LoopRec reuses a weight-shared backbone block and makes an exit-or-continue
decision for each item at each recursion boundary, enabling item-level
adaptive depth for sequential recommendation.

## Contents

```text
src/main.py                           Training and evaluation entry point
src/models/LoopRec.py                  LoopRec model
src/utils/looprec.py                   Recursive routing and ordered-exit logic
src/utils/looprec_backbone_adapter.py  Backbone adapters for LoopRec
src/models/{SASRec,BSARec,NextItNet,LRURec}.py
src/scripts/config/                   Model configuration templates
datasets/scripts/amazon-preprocess-abs.py
tests/                                Tests
```

Supported backbones:

- SASRec
- BSARec
- NextItNet
- LRURec

Datasets:

- `Software`
- `Office_Products`
- `Patio_Lawn_and_Garden`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install a PyTorch build matching your CUDA version.

## Data

See [datasets/README.md](datasets/README.md). After preprocessing, the default
expected data root is:

```text
datasets/processed_clean_v3_rearec/
```

You can also point to an external processed-data root:

```bash
export DATA_PATH_OVERRIDE=/path/to/processed_clean_v3_rearec
```

## Quick Start

Run a fixed-depth SASRec baseline:

```bash
bash src/scripts/run_train.bash sasrec Software 0
```

Run LoopRec with a SASRec backbone:

```bash
bash src/scripts/run_train.bash looprec_sasrec Software 0
```

Run LoopRec with transfer backbones:

```bash
bash src/scripts/run_train.bash looprec_bsarec Office_Products 0
bash src/scripts/run_train.bash looprec_nextitnet Patio_Lawn_and_Garden 0
bash src/scripts/run_train.bash looprec_lrurec Software 0
```

The launcher reads `src/scripts/config/<model_id>.conf` and takes the dataset
from the command line. See [configuration options](src/scripts/config/README.md)
for environment-variable overrides.

## Tests

```bash
pytest -q
```

The tests use synthetic inputs and do not require downloaded datasets.
