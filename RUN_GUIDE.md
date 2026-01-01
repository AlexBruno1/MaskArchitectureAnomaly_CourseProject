# Usage Guide for EoMT Anomaly Detection Scripts

This document explains how to run the key scripts for training and evaluating the EoMT model for anomaly detection.

## 1. `run_eval_experiment.py`

This is the high-level automation script for evaluating a trained checkpoint across multiple datasets (e.g., RoadAnomaly, FS LostFound, etc.).

### Usage
```bash
python run_eval_experiment.py --ckpt_path <path_to_checkpoint> --dataset_root <path_to_datasets> [options]
```

### Arguments
| Argument | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `--ckpt_path` | Yes | - | Path to the model checkpoint (`.ckpt` or `.bin`). |
| `--dataset_root` | Yes | - | Root directory containing the dataset folders (e.g. `Validation_Dataset`). |
| `--result_dir` | No | `result` | Directory where results (`.txt` logs and `.csv` summary) will be saved. |
| `--img_height` | No | `640` | Height to resize images to during evaluation. |
| `--img_width` | No | `640` | Width to resize images to during evaluation. |

### Example
```bash
python run_eval_experiment.py --ckpt_path trained_models/model.ckpt --dataset_root ../Validation_Dataset
```

---

## 2. `eval/evalAnomalyEomt.py`

This is the core evaluation script processing a single dataset/folder. It computes anomaly scores (MSP, MaxLogit, MaxEntropy) and metrics (AuPRC, FPR@95).

### Usage
```bash
python eval/evalAnomalyEomt.py --ckpt_path <path_to_checkpoint> --input <glob_pattern> [options]
```

### Key Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--input` | *See script* | Glob pattern or list of input image paths to evaluate. |
| `--ckpt_path` | **Required** | Path to the trained model checkpoint. |
| `--result_file` | `results_eomt.txt` | File path to append results to. |
| `--img_height`, `--img_width` | `640` | Image resolution for inference. |
| `--cpu` | `False` | Force CPU inference if set. |
| `--num_classes` | `19` | Number of semantic classes (default for Cityscapes). |
| `--backbone_name` | `vit_base_patch14_reg4_dinov2` | Name of the ViT backbone. |

### Example
```bash
python eval/evalAnomalyEomt.py --ckpt_path trained_models/model.ckpt --input "../Validation_Dataset/RoadAnomaly/images/*.jpg" --result_file results.txt
```

---

## 3. `eomt/train_net_lora.py`

This script handles training (fine-tuning) the EoMT model using LoRA (Low-Rank Adaptation) and Logit Normalization. It uses PyTorch Lightning CLI.

### Usage
```bash
python eomt/train_net_lora.py fit --config <config.yaml> [options]
```

### LoRA & Training Arguments
These arguments can be passed via command line or defined in limits config file under the `fit` subcommand.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--lora_rank` | int | `32` | Rank of LoRA adapters. |
| `--lora_alpha` | int | `64` | LoRA alpha scaling factor. |
| `--lora_dropout` | float | `0.0` | Dropout probability for LoRA layers. |
| `--lora_targets` | list | `["qkv", "q_proj", "v_proj"]` | List of module names to inject LoRA into. |
| `--logit_norm_temperature` | float | `0.04` | Temperature for Logit Normalization loss. |
| `--head_only` | flag | `False` | If set, freezes the encoder and trains only classification heads. |
| `--activation_checkpointing` | flag | `False` | Enables gradient checkpointing to save memory. |
| `--pretrained_path` | str | `None` | Path to initialize weights from before training starts. |

#### Note:
- If you use a different lora_alpha then 64, you should also change the lora_alpha in eomt\training\utils\checkpoint.py line 74

### Standard Lightning CLI Arguments
You also need to specify the model and data configuration:
- `--trainer.max_epochs`: Number of epochs.
- `--data.init_args.img_size`: Image size for training (e.g. `[640, 640]`).
- `--data.init_args.batch_size`: Batch size per GPU.
- `--trainer.accumulate_grad_batches`: Gradient accumulation steps.
- `--data.init_args.path`: Path to the dataset.

### Example Command
```bash
python train_net_lora.py fit `
  --config configs/dinov2/cityscapes/semantic/eomt_base_640.yaml `
  --head_only `
  --pretrained_path="../trained_eomt/eomt_cityscapes.bin" `
  --trainer.max_epochs=20 `
  --trainer.check_val_every_n_epoch 20 `
  --data.init_args.img_size="[1024, 1024]" `
  --data.init_args.batch_size=4 `
  --trainer.accumulate_grad_batches=4 `
  --data.init_args.path="cityscapes" --logit_norm_temperature 1.2
```
