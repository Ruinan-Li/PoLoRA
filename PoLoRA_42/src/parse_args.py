import argparse
import os
import sys

# Add src directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(
    description="Parser For Arguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# paths
parser.add_argument("-save_path", dest="save_path", default="./checkpoint/", help="Path of saved models")
parser.add_argument("-log_path", dest="log_path", default="./logs/", help="Path of saved logs")
parser.add_argument("-data_path", dest="data_path", default="./data/", help="Path of dataset")

# global setting
parser.add_argument("-random_seed", dest="random_seed", type=int, default=42, help="Set random seeds")
parser.add_argument("-dataset", dest="dataset", default="HYBRID", help="dataset name")
parser.add_argument("-gpu", dest="gpu", default=0, help="number of gpu")

# model setting
parser.add_argument("-model_name", dest="model_name", default="LoraKGE", help="name of model")
parser.add_argument("-batch_size", dest="batch_size", default=256, help="Set the batch size")
parser.add_argument("-epoch_num", dest="epoch_num", default=100, help="Set the epoch (deprecated, use epoch_num_snapshot0 or epoch_num_snapshot1plus)")
parser.add_argument("-epoch_num_snapshot0", dest="epoch_num_snapshot0", default=200, help="Set the epoch number for snapshot 0")
parser.add_argument("-epoch_num_snapshot1plus", dest="epoch_num_snapshot1plus", default=200, help="Set the epoch number for snapshots >= 1")
parser.add_argument("-note", dest="note", default="", help="The note of log file name")
parser.add_argument("-snapshot_num", dest="snapshot_num", default=5, help="The number of snapshots")
parser.add_argument("-neg_ratio", dest="neg_ratio", default=10, help="the ratio of negtive/postive facts")
parser.add_argument("-train_new", dest="train_new", type=lambda x: str(x).lower() == "true", default=True, help="True: train on new facts only; False: train on all seen facts")
parser.add_argument("-valid_metrics", dest="valid_metrics", default="mrr")
parser.add_argument("-patience", dest="patience", default=8, help="early stop step")

# new updates
parser.add_argument("-debug", dest="debug", default=False, help="test mode")
parser.add_argument("-record", dest="record", default=False, help="Record the loss of different layers")
parser.add_argument("-predict_result", dest="predict_result", default=False, help="The result of predict")
parser.add_argument("-ent_r", dest="ent_r", default=40, help="The rank of ent lora")
parser.add_argument("-rel_r", dest="rel_r", default=40, help="The rank of rel lora")
parser.add_argument("-use_poincare_eval", dest="use_poincare_eval", type=lambda x: str(x).lower() == "true", default=True, help="Use Poincare scoring for evaluation")
# poincare snapshot-0 setting
parser.add_argument("-poincare_dim", dest="poincare_dim", default=40, help="embedding dim for MuRP/MuRE")
parser.add_argument("-poincare_lr", dest="poincare_lr", default=50.0, help="learning rate for snapshot-0 MuRP optimizer")
parser.add_argument("-poincare_nneg", dest="poincare_nneg", default=50, help="num of negative samples for MuRP")
parser.add_argument("-poincare_batch_size", dest="poincare_batch_size", default=128, help="batch size for MuRP")
parser.add_argument("-poincare_model", dest="poincare_model", default="poincare", help="choose poincare(MuRP) or euclidean(MuRE)")
parser.add_argument("-lora_lr", dest="lora_lr", default=1.5, help="learning rate for snapshots >=1 (Riemannian LoRA)")
# Euclidean scheduler settings
parser.add_argument("-euclid_scheduler", dest="euclid_scheduler", type=lambda x: str(x).lower() == "true", default=True, help="Enable cosine warm restarts for Euclidean params")
parser.add_argument("-euclid_scheduler_T0", dest="euclid_scheduler_T0", type=int, default=200, help="T0 for cosine warm restarts")
parser.add_argument("-euclid_scheduler_Tmult", dest="euclid_scheduler_Tmult", type=int, default=2, help="Tmult for cosine warm restarts")
parser.add_argument("-euclid_scheduler_eta_min", dest="euclid_scheduler_eta_min", type=float, default=0.1, help="Minimum lr for cosine warm restarts")
# PER settings
parser.add_argument("-per_enable", dest="per_enable", type=lambda x: str(x).lower() == "true", default=True, help="Enable mixed priority experience replay")
parser.add_argument("-per_replay_ratio", dest="per_replay_ratio", type=float, default=1.5, help="PER: ratio of old to new samples (old_samples = new_samples × per_replay_ratio). Auto-computed if not specified")
parser.add_argument("-per_eps", dest="per_eps", default=1e-6, help="PER: epsilon to prevent zero")
parser.add_argument("-per_alpha", dest="per_alpha", default=0.5, help="PER: |loss| exponent")
parser.add_argument("-per_gamma", dest="per_gamma", default=0.5, help="PER: structural importance exponent")
parser.add_argument("-per_beta", dest="per_beta", default=0.5, help="PER: IS weight exponent")

args = parser.parse_args()

# Auto-compute per_replay_ratio 
if args.per_replay_ratio is None and args.per_enable:
    try:
        from replay.adaptive_replay_ratio import get_adaptive_replay_ratio
        data_path = os.path.join(args.data_path, args.dataset)
        if os.path.exists(data_path):
            args.per_replay_ratio = get_adaptive_replay_ratio(args.dataset, args.data_path)
            print(f"[Adaptive Replay] Automatically computed replay ratio for {args.dataset}: {args.per_replay_ratio:.3f}")
        else:
            print(f"[Warning] Dataset path not found: {data_path}, using default replay ratio 0.2")
            args.per_replay_ratio = 0.2
    except Exception as e:
        print(f"[Warning] Failed to compute adaptive replay ratio: {e}, using default 0.2")
        args.per_replay_ratio = 0.2
elif args.per_replay_ratio is None:
    args.per_replay_ratio = 1.5  
