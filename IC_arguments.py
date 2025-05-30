import argparse

def arguments():
    parser = argparse.ArgumentParser(description="Image Classification with NAS")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224",
                        choices=["google/vit-base-patch16-224", "google/vit-base-patch16-224-in21k", "microsoft/swin-tiny-patch4-window7-224",
                                 "microsoft/swin-base-patch4-window7-224", "microsoft/swin-large-patch4-window7-224",
                                 "facebook/deit-tiny-patch16-224", "facebook/deit-small-patch16-224",
                                 "facebook/deit-base-patch16-224", "facebook/deit-base-patch16-224-distilled",],
                        help="Model name or path")
    parser.add_argument("--processor_name", type=str, default="google/vit-base-patch16-224",
                        choices=["google/vit-base-patch16-224", "google/vit-base-patch16-224-in21k", "microsoft/swin-tiny-patch4-window7-224",
                                 "microsoft/swin-base-patch4-window7-224", "microsoft/swin-large-patch4-window7-224",
                                 "facebook/deit-tiny-patch16-224", "facebook/deit-small-patch16-224",
                                 "facebook/deit-base-patch16-224", "facebook/deit-base-patch16-224-distilled"],
                        help="Processor name or path")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "imagenet-1k", "slegroux/tiny-imagenet-200-clean", "zh-plus/tiny-imagenet"],
                        help="Dataset name")
    parser.add_argument("--cache_dir", type=str, default="/ptmp/LAS/msamani/.cache",
                        help="Directory to cache datasets and models")
    parser.add_argument("--log_dir", type=str, default="/work/LAS/jannesar-lab/msamani/SuperSAM/logs",
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
    parser.add_argument("--trainable", type=str, default="em",
                        choices=["e", "m", "p"],
                        help="Trainable parameters for sam: e=vision_encoder, m=mask_encoder, p=prompt_encoder" \
                             "Trainable parameters for vit(IC): e=vit, m=classifier, p=vit.embedding")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N steps")
    parser.add_argument("--no_verbose", action="store_true",
                        help="Disable verbose logging")
    parser.add_argument("--reorder", type=str, default="per_epoch",
                        choices=["once", "per_epoch", "per_batch", "none"],
                        help="When to reorder MLP layers")
    parser.add_argument("--reorder_method", type=str, default="wanda",
                        choices=["magnitude", "wanda", "none"],
                        help="Method for MLP layer reordering")
    parser.add_argument("--sandwich", type=str, default="lsm",
                        choices=["l", "s", "m", "ls", "lm", "sm", "lsm"],
                        help="Sandwich configuration for training (l:largest, s:smallest, m:medium)")
    parser.add_argument("--huggingface_token", type=str, default="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",
                        help="Hugging Face token for ImageNet-1k dataset")
    parser.add_argument("--subsample", type=bool, default=False,
                        help="Subsample ImageNet-1k dataset")
    parser.add_argument("--tensorboard_visual", type=bool, default=False,
                        help="Enable TensorBoard visualization after saving checkpoints")
    return parser.parse_args()
