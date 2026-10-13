# SAM3 fine-tuning template

Portable text-query and instance-mask supervised fine-tuning, extracted from a working SAM3 training workflow. No datasets, weights, private paths, cluster accounts or evaluation results are included.

## Setup

Use Python 3.11+, a CUDA-enabled PyTorch/torchvision pair compatible with your GPU, and the official SAM3 prerequisites. Obtain `sam3.pt` through the official access/license process.

```bash
python scripts/bootstrap.py
pip install -e 'vendor/sam3[train]'
pip install -r requirements.txt
```

The bootstrap pins official SAM3 to `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7` and applies `patches/training.patch`. It refuses dirty or mismatched checkouts. The patch retains freeze policies, frozen-backbone forward handling, activation checkpointing, query microbatching, gradient accumulation and large-loss memory safeguards from the original training implementation. Upstream code and derived files remain subject to the SAM3 license; see `LICENSE.sam3`.

## Data

Create a JSON file with `format: sam3_sft_queries_v1` and an `images` array:

```json
{
  "format": "sam3_sft_queries_v1",
  "images": [{
    "id": 1, "file_name": "example.jpg", "width": 2, "height": 2,
    "objects": [{"id": 7, "bbox": [0, 0, 2, 2],
      "segmentation": {"size": [2, 2], "counts": [0, 4]}}],
    "queries": [
      {"text": "square", "object_ids": [7], "is_exhaustive": true},
      {"text": "circle", "object_ids": [], "is_exhaustive": true}
    ]
  }]
}
```

This example is synthetic. Replace it with real images and masks. Boxes use **pixel xywh**; masks use COCO RLE (`size=[height,width]`, compressed string or uncompressed counts). Object IDs are image-local. Parent/leaf labels, if desired, are independent queries referencing the same object IDs. An empty target list means confirmed absence, never a missing annotation. The loader rejects invalid boxes, duplicate IDs and dangling query references. Keep train/validation/test source-disjoint. This in-memory JSON adapter is a template for moderate datasets; production tar/offset conversion and taxonomy construction are intentionally excluded.

## Train

```bash
export SAM3_CHECKPOINT=/path/to/sam3.pt
export SAM3_DATA_ROOT=/path/to/images
export SAM3_ANNOTATION_FILE=/path/to/train.json
export SAM3_RUN_DIR=/path/to/new_run
GPUS_PER_NODE=8 bash scripts/train.sh
```

Run a bounded smoke first: `SAM3_LIMIT_IMAGES=64 SAM3_MAX_EPOCHS=1 SAM3_RUN_DIR=/path/to/smoke bash scripts/train.sh`. Supply at least one local batch per rank; tiny smoke datasets need fewer GPUs. Use a new run directory for a new experiment; an existing checkpoint in that directory triggers the upstream resume behavior.

Defaults: TFDM (vision frozen; text, fusion, decoder and instance-mask head trainable), BF16, activation checkpointing, local batch 1, accumulation 1, four epochs, seed 123. Transformer/text base LR: `1.6e-4 / 1e-5`; inverse-square-root timescale 200, warmup/cooldown 20 steps. These LRs originated from a 32-GPU experiment: **global batch = world size × local batch × accumulation**; retune for other sizes and datasets. Override `SAM3_MAX_EPOCHS`, `SAM3_LOCAL_BATCH`, `SAM3_GRAD_ACCUM_STEPS`, `SAM3_FREEZE_POLICY` (`fdm/tfdm/vfdm/full`), `SAM3_SCHEDULER_TIMESCALE`, `SAM3_LR_TRANSFORMER`, `SAM3_LR_TEXT`, `SAM3_WORKERS`, or edit `configs/train.yaml`.

For multi-node training, launch once per node with the same environment and `NNODES`, `GPUS_PER_NODE`, `MASTER_ADDR`, `MASTER_PORT`, plus a unique `NODE_RANK`. A scheduler can allocate nodes; no site-specific Slurm account/reservation is assumed.

Each epoch saves `checkpoints/checkpoint_N.pt`, including model, optimizer, scaler and step state, plus the latest `checkpoint.pt`. Logs and TensorBoard events are written under the run directory. Validation is disabled in the template; select hyperparameters on a separate validation set and reserve test benchmarks for final reporting.

## Checks and limits

```bash
python -m unittest discover -s tests
```

With the training environment active, `SAM3_VALIDATE_CONFIG=1 bash scripts/train.sh` prints the resolved configuration without training. The patch and schema tests were checked during extraction; the **new portable template has not yet had an end-to-end GPU smoke**. The original implementation was trained on A100 GPUs. Verify your CUDA stack, masks and positive/negative supervision before a full run. This repository provides training only; benchmark-specific evaluation and human-review workflows remain separate.
