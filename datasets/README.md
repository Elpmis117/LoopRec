# Datasets

The experiments use the following Amazon Reviews 2023 categories with the
`amazon_abs_v3_rearec` preprocessing profile:

- `Software`
- `Office_Products`
- `Patio_Lawn_and_Garden`

Download the category JSONL files from Amazon Reviews 2023 and arrange them as
follows:

```text
datasets/raw/
  Software/Software.jsonl
  Office_Products/Office_Products.jsonl
  Patio_Lawn_and_Garden/Patio_Lawn_and_Garden.jsonl
```

Build the processed data in two steps:

```bash
python datasets/scripts/amazon-preprocess-abs.py \
  --profile amazon_abs_v1 \
  --dataset Software Office_Products Patio_Lawn_and_Garden \
  --raw-root datasets/raw \
  --output-root datasets/processed_clean

python datasets/scripts/amazon-preprocess-abs.py \
  --profile amazon_abs_v3_rearec \
  --dataset Software Office_Products Patio_Lawn_and_Garden \
  --canonical-input-root datasets/processed_clean \
  --output-root datasets/processed_clean_v3_rearec
```

Training configs default to `datasets/processed_clean_v3_rearec`.
