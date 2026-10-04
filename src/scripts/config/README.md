# Training Configs

This directory contains model configuration templates for training and evaluation.

Supported datasets:

- `Software`
- `Office_Products`
- `Patio_Lawn_and_Garden`

Supported model ids:

- `sasrec`
- `bsarec`
- `nextitnet`
- `lrurec`
- `looprec_sasrec`
- `looprec_bsarec`
- `looprec_nextitnet`
- `looprec_lrurec`

Each file is a small Bash config sourced by:

```bash
bash src/scripts/run_train.bash looprec_sasrec Software 0
```

The launcher takes the dataset as its second argument.

Common overrides:

```bash
DATA_PATH_OVERRIDE=/path/to/processed_clean_v3_rearec
LR_OVERRIDE=0.0006
BATCH_SIZE_OVERRIDE=1024
LOOPREC_WARMUP_EPOCHS_OVERRIDE=12
```

The default data profile is `amazon_abs_v3_rearec`, matching the
Amazon absolute-time preprocessing protocol.
