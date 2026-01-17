# PoLoRA: Poincar´e LoRA with Adaptive Structural Replay for Temporal Knowledge Graph Reasoning

PoLoRA is a method for continual knowledge graph embedding that leverages Low-Rank Adaptation (LoRA) in Poincare hyperbolic space. It enables efficient incremental learning on evolving knowledge graphs while maintaining the hierarchical structure.

## Features

- **Riemannian LoRA**: Implements LoRA entirely on Poincare ball 
- **Continual Learning**: Supports multiple snapshots with incremental updates
- **Hyperbolic Geometry**: Uses MuRP (Multi-Relational Poincare) scoring function for better hierarchical representation
- **Priority Experience Replay (PER)**: Adaptive replay mechanism with structural importance weighting
- **Efficient Adaptation**: Low-rank parameter updates reduce memory and computational costs

## Framework

The method operates in two phases:
- **Snapshot 0**: Full Poincare embedding training using MuRP
- **Snapshots ≥1**: Incremental updates using Riemannian LoRA on Poincare ball



### Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
PoLoRA/
├── checkpoint/          # Saved model checkpoints
├── data/               # Dataset files 
├── logs/               # Training logs
├── src/
│   ├── data_load/      # Data loading modules
│   │   ├── KnowledgeGraph.py
│   │   └── data_loader.py
│   ├── model/          # Model definitions
│   │   ├── BaseModel.py
│   │   ├── LoraKGE.py
│   │   └── model_process.py
│   ├── poincare/       # Poincare geometry operations
│   │   ├── load_data.py
│   │   ├── mapping.py
│   │   ├── model.py
│   │   ├── riemann_lora.py
│   │   ├── rsgd.py
│   │   ├── snapshot0.py
│   │   └── utils.py
│   ├── replay/         # Experience replay modules
│   │   ├── adaptive_replay_ratio.py
│   │   └── per_buffer.py
│   ├── parse_args.py
│   ├── train.py
│   ├── test.py
│   └── utils.py
├── main.py             
├── main.sh             
└── README.md
```

## Usage

### Data Preparation

Prepare your dataset in the following structure:

```
data/
└── DATASET_NAME/
    ├── 0/
    │   ├── train.txt
    │   ├── valid.txt
    │   ├── test.txt
    │   ├── entity2id.txt
    │   ├── relation2id.txt
    │   └── train_edges_betweenness.txt  # For PER
    ├── 1/
    │   └── ...
    └── ...
```

Each `train.txt`, `valid.txt`, `test.txt` should contain triples in format:
```
head_entity relation tail_entity
```

### Key Parameters

#### Paths
- `-data_path`: Path to dataset directory (default: `./data/`)
- `-save_path`: Path to save checkpoints (default: `./checkpoint/`)
- `-log_path`: Path to save logs (default: `./logs/`)

#### Model Settings
- `-model_name`: Model name (default: `LoraKGE`)
- `-batch_size`: Batch size (default: `256`)
- `-snapshot_num`: Number of snapshots (default: `5`)
- `-epoch_num_snapshot0`: Epochs for snapshot 0 (default: `50`)
- `-epoch_num_snapshot1plus`: Epochs for snapshots ≥1 (default: `50`)
- `-neg_ratio`: Negative sampling ratio (default: `10`)

#### LoRA Settings
- `-ent_r`: LoRA rank for entities (default: `40`)
- `-rel_r`: LoRA rank for relations (default: `40`)
- `-lora_lr`: Learning rate for LoRA (default: `1.5`)

#### Poincare Settings
- `-poincare_dim`: Embedding dimension (default: `40`)
- `-poincare_lr`: Learning rate for snapshot 0 (default: `50.0`)
- `-poincare_batch_size`: Batch size for snapshot 0 (default: `128`)
- `-poincare_nneg`: Negative samples for snapshot 0 (default: `50`)

#### PER (Priority Experience Replay) Settings
- `-per_enable`: Enable PER (default: `True`)
- `-per_replay_ratio`: Replay ratio (auto-computed if not specified)
- `-per_alpha`: Loss exponent (default: `1`)
- `-per_gamma`: Structural importance exponent (default: `0`)
- `-per_beta`: IS weight exponent (default: `0`)

#### Other Settings
- `-train_new`: Train on new facts only (default: `True`)
- `-patience`: Early stopping patience (default: `8`)
- `-random_seed`: Random seed (default: `42`)

### Example Training Command

```bash
python main.py \
    -dataset HYBRID \
    -gpu 0 \
    -snapshot_num 5 \
    -epoch_num_snapshot0 200 \
    -epoch_num_snapshot1plus 200 \
    -ent_r 40 \
    -rel_r 40 \
    -poincare_dim 40 \
    -per_enable True \
    -batch_size 256
```

### Quick Start

To quickly start training, run `main.sh`:

```bash
bash main.sh
```

