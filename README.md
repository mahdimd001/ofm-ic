# SuperViT-NAS: Crafting Efficient Vision Transformers for Image Classification through Neural Architecture Search

SuperViT-NAS is a Neural Architecture Search (NAS) framework for crafting efficient Vision Transformers (ViTs) for image classification. It leverages structured pruning and parameter prioritization to create a supernetwork, optimizing performance on datasets like CIFAR-10, CIFAR-100, and ImageNet-1k.

## Features

- **Supported Datasets**: CIFAR-10, CIFAR-100, and ImageNet-1k (with optional subsampling: 5,000 train, 1,000 validation images).
- **Model**: Vision Transformer (ViT) based on `google/vit-base-patch16-224-in21k`.
- **NAS**: Uses Once-for-All (OFA) supernetwork with elastic configurations for attention, hidden layers, and residuals.
- **MLP Reordering**: Supports `magnitude` and `wanda` methods, applied `once`, `per_epoch`, `per_batch`, or `none`.
- **Training Configurations**:
  - Sandwich training: Largest (`l`), Smallest (`s`), Medium (`m`) submodels.
  - Trainable parameters: ViT (`e`), classifier (`m`), embeddings (`p`).
- **Checkpointing & Resuming**: Saves checkpoints per epoch
- **Visualization**: TensorBoard with 3-column grids (image, predicted class, ground truth).
- **Performance**:
  - CIFAR-10: \~96% accuracy.
  - CIFAR-100: \~76-85% accuracy.
  - ImageNet-1k: \~0% (subset) or \~0% (full).
  - FLOPs: \~25.6-35.7 GFLOPs.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/mahdimd001/ofm-ic.git
   cd ofm-ic-main
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install torch torchvision transformers datasets ptflops tensorboard matplotlib>=3.5 numpy
   ```

3. Authenticate for ImageNet-1k (if needed):

   - Get a Hugging Face token: https://huggingface.co/settings/tokens

   - Set environment variable or pass via `--huggingface_token`:

     ```bash
     export HF_TOKEN=<your_token>
     ```

## Usage

change the arguments in "IC_arguments.py" and then::

### CIFAR-10

```bash
python IC_nas.py
```



## Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--model_name` | Model name or path | `google/vit-base-patch16-224-in21k` |
| `--processor_name` | Processor name or path | `google/vit-base-patch16-224` |
| `--dataset` | Dataset: `cifar10`, `cifar100`, `imagenet-1k` | `cifar100` |
| `--cache_dir` | Directory to cache datasets and models | `/lustre/.../.cache` |
| `--batch_size` | Batch size for training and evaluation | `32` |
| `--epochs` | Number of training epochs | `50` |
| `--lr` | Learning rate | `5e-5` |
| `--weight_decay` | Weight decay | `0.0` |
| `--scheduler_type` | Scheduler: `linear`, `cosine`, `constant` | `linear` |
| `--trainable` | Trainable parameters: `e` (ViT), `m` (classifier), `p` (embeddings) | `em` |
| `--save_interval` | Save checkpoint every N steps | `10` |
| `--no_verbose` | Disable verbose logging | `False` |
| `--reorder` | Reorder MLP layers: `once`, `per_epoch`, `per_batch`, `none` | `per_none` |
| `--reorder_method` | Reordering method: `magnitude`, `wanda`, `none` | `magnitude` |
| `--sandwich` | Submodel training: `l` (largest), `s` (smallest), `m` (medium) | `lsm` |
| `--huggingface_token` | Hugging Face token for ImageNet-1k | `hf_lYSBWtf...` |
| `--subsample` | Subsample ImageNet-1k (5,000 train, 1,000 validation) | `True` |

## Output

- **Logs**: Stored in `logs/<timestamp>_dataset[...]/experiment_log-<timestamp>.log`.

- **Checkpoints**: Saved in `logs/<timestamp>_dataset[...]/checkpoints/epoch_X.pth` and `best_smallest.pth`.

- **TensorBoard**: Visualize predictions in `logs/<timestamp>_dataset[...]/tensorboard/`:

  ```bash
  tensorboard --logdir ./logs
  ```

  Open `http://localhost:6006`.

## Expected Results

- **CIFAR-10**: \~96% accuracy, \~50.6 GFLOPs, classifier: 7,690 params.
- **CIFAR-100**: \~80-90% accuracy, \~50.62 GFLOPs, classifier: 76,900 params.
- **ImageNet-1k (Subset)**: \~70-80% accuracy, \~50.7 GFLOPs, classifier: 769,000 params.
- **ImageNet-1k (Full)**: \~80-85% accuracy, \~50.7 GFLOPs, classifier: 769,000 params.
- **Total Parameters**: \~85,806,346 (CIFAR-10/CIFAR-100), \~86,798,440 (ImageNet-1k).

## Troubleshooting

- **ImageNet-1k Authentication**:

  ```bash
  huggingface-cli login
  ```



- **Low Accuracy**:

  - Adjust `--lr` (e.g., `1e-4`), increase `--epochs` (e.g., `150`).
  - Use stronger pre-trained weights: `google/vit-base-patch16-224`.

- **Checkpoint Issues**:

  ```bash
 NA
  ```

## Contributing

NA

## License

NA

## Citation

If you use SuperViT-NAS in your research, please cite:

```
NA
```

---

Replace `yourusername` with your GitHub username and update the `author` field in the citation with your name. If you have a specific license file or additional details, include them in the README.