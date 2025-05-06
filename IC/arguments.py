import argparse

def arguments():
    parser = argparse.ArgumentParser(description="Image Classification with NAS")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224-in21k",
                        help="Model name or path")
    parser.add_argument("--processor_name", type=str, default="google/vit-base-patch16-224",
                        help="Processor name or path")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "imagenet-1k"],
                        help="Dataset name")
    parser.add_argument("--cache_dir", type=str, default="/work/LAS/jannesar-lab/msamani/.cache",
                        help="Directory to cache datasets and models")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--scheduler_type", type=str, default="linear",
                        choices=["linear", "cosine", "constant"],
                        help="Learning rate scheduler type")
    parser.add_argument("--trainable", type=str, default="lm",
                        help="Trainable components (l:largest, m:medium, s:smallest)")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N steps")
    parser.add_argument("--no_verbose", action="store_true",
                        help="Disable verbose logging")
    parser.add_argument("--reorder", type=str, default="per_epoch",
                        choices=["once", "per_epoch", "per_batch", "none"],
                        help="When to reorder MLP layers")
    parser.add_argument("--reorder_method", type=str, default="magnitude",
                        choices=["magnitude", "wanda", "none"],
                        help="Method for MLP layer reordering")
    return parser.parse_args()