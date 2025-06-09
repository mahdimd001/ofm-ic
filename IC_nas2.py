import os
import torch
from datasets import load_dataset, Image as DatasetImage
from IC_arguments import arguments
from logger import init_logs, get_logger
import functools
import numpy as np
import sys
from huggingface_hub import login
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers.models.vit.modeling_vit import ViTSelfAttention
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm
from thop import profile
import datetime
import copy
import logging
from huggingface_hub import login
import evaluate
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Custom ViT Self Attention with reduced heads
class PrunedViTSelfAttention(ViTSelfAttention):
    def __init__(self, config, num_heads):
        # Initialize with original config but override num_attention_heads
        super().__init__(config)
        self.num_attention_heads = num_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Recreate linear layers with correct dimensions
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

# Function to calculate parameters
def calculate_params(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
    return total_params / 1e6  # In millions

# Function to compute FLOPs
def compute_flops(model):
    """Compute forward pass FLOPs for the model in GFLOPs, dataset-agnostic."""
    model.eval()
    model = model.to('cpu')


    height = width = 224
    input_shape = (1, 3, height, width)
    #self.logger.info(f"Computing forward FLOPs with input shape: {input_shape}")

    # Compute FLOPs and parameters using thop
    input_tensor = torch.randn(input_shape)
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = macs
    gflops = flops / 1e9

    #self.logger.info(f"Total params: {params:,}, Forward GFLOPs: {gflops:.2f}")

    return params, gflops

# Function to compute head scores
def get_head_scores_QKV(model):
    # Get the norms of query, key, and value weights for each head in each layer
    # Score
    scores = []
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    for layer_idx, layer in enumerate(model.vit.encoder.layer):
        q_weight = layer.attention.attention.query.weight
        k_weight = layer.attention.attention.key.weight
        v_weight = layer.attention.attention.value.weight
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        qkv_weight = qkv_weight.view(3, model.config.num_attention_heads, head_dim, model.config.hidden_size)
        qkv_weight = qkv_weight.permute(1, 0, 2, 3).reshape(model.config.num_attention_heads, -1)
        head_norms = torch.norm(qkv_weight, dim=1).tolist()
        scores.append(head_norms)
    return scores


def get_head_scores_Output(model):
    # Get the norms of output dense weights for each head in each layer
    # Score2,3
    scores = []
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    for layer_idx, layer in enumerate(model.vit.encoder.layer):
        W_output = layer.attention.output.dense.weight  # shape (hidden_size, hidden_size)
        layer_scores = []
        for h in range(model.config.num_attention_heads):
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim
            submatrix = W_output[:, start_idx:end_idx]
            score = torch.norm(submatrix, p='fro').item()
            layer_scores.append(score)
        scores.append(layer_scores)
    return scores

def collect_head_activations(model, data_loader, device, max_samples=100):
    model.eval()

    model = model.to(device)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_layers = len(model.vit.encoder.layer)
    num_heads = model.config.num_attention_heads
    activation_norms = [[0.0] * num_heads for _ in range(num_layers)]
    sample_count = 0

    hooks = []
    def hook_fn(module, input, output, layer_idx):
        if isinstance(output, tuple):
            attention_output = output[0].detach()
        else:
            attention_output = output.detach()
        
        if len(attention_output.shape) == 2:
            attention_output = attention_output.unsqueeze(0)
        
        if len(attention_output.shape) != 3:
            raise ValueError(f"Unexpected attention output shape: {attention_output.shape}")
        
        batch_size, seq_len, hidden_size = attention_output.shape
        attention_output = attention_output.view(batch_size, seq_len, num_heads, head_dim)
        for h in range(num_heads):
            head_output = attention_output[:, :, h, :]  # (batch_size, seq_len, head_dim)
            norm = torch.norm(head_output, p=2).item()  # Norm across all dimensions
            activation_norms[layer_idx][h] += norm
        nonlocal sample_count
        sample_count += batch_size

    for layer_idx, layer in enumerate(model.vit.encoder.layer):
        hook = layer.attention.output.dense.register_forward_hook(
            lambda m, i, o, idx=layer_idx: hook_fn(m, i, o, idx)
        )
        hooks.append(hook)

    try:
        with torch.no_grad():
            for inputs in tqdm(data_loader, desc="Collecting activations"):
                images = inputs["pixel_values"].to(device)

                if images.dim() == 3:
                    images = images.unsqueeze(0)

                model(images)
                sample_count += images.size(0)

                if sample_count >= max_samples:
                    break

            if sample_count == 0:
                raise ValueError("No samples processed. Check data_loader.")
    except Exception as e:
        print(f"Error during activation collection: {e}")
        raise
    finally:
        for hook in hooks:
            hook.remove()


    for layer_idx in range(num_layers):
        for h in range(num_heads):
            activation_norms[layer_idx][h] /= sample_count if sample_count > 0 else 1

    return activation_norms

def get_head_scores_OutputActivation(model, data_loader, device, cache_dir="./cache", max_samples=100):
    # Collect activation norms for each head in each layer
    # Score4
    # wanda activation norms
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "wanda_activation_norms.pth")
    
    if os.path.exists(cache_path):
        print(f"Loading cached activation norms from {cache_path}...")
        try:
            activation_norms = torch.load(cache_path, map_location=device)
        except Exception as e:
            print(f"Error loading cached activation norms: {e}")
            raise
    else:
        activation_norms = collect_head_activations(model, data_loader, device, max_samples)
        try:
            torch.save(activation_norms, cache_path)
            print(f"Saved activation norms to {cache_path}")
        except Exception as e:
            print(f"Error saving activation norms: {e}")
            raise

    scores = []
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    for layer_idx, layer in enumerate(model.vit.encoder.layer):
        W_output = layer.attention.output.dense.weight
        layer_scores = []
        for h in range(model.config.num_attention_heads):
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim
            submatrix = W_output[:, start_idx:end_idx]
            weight_norm = torch.norm(submatrix, p='fro').item()
            activation_norm = activation_norms[layer_idx][h]
            score = weight_norm * activation_norm
            layer_scores.append(score)
        scores.append(layer_scores)
    return scores




def get_head_scores_Paper(model, data_loader, device, num_batches=10):
    """
    Compute head importance scores based on the gradient-based method from
    'Are Sixteen Heads Really Better than One?'.
    
    Args:
        model: Pre-trained ViT model.
        data_loader: DataLoader with input data.
        device: Device to run computation on (e.g., 'cuda').
        num_batches: Number of batches to process (default: 10).
    
    Returns:
        List of tensors, one per layer, each of shape (num_heads,), containing importance scores.
    """
    model = model.to(device)
    model.train()  # Enable gradients

    num_layers = len(model.vit.encoder.layer)
    num_heads = model.config.num_attention_heads
    # Accumulators for importance scores per layer
    importance_accumulator = [torch.zeros(num_heads, device=device) for _ in range(num_layers)]
    total_samples = 0

    # Hook function to compute importance when gradient is available
    def hook_fn(grad, context_layer, layer_idx):
        # grad and context_layer: (batch_size, num_heads, seq_len, head_dim)
        # Compute per-sample contribution: sum over seq_len and head_dim
        contribution = (context_layer * grad).sum(dim=[2, 3])  # (batch_size, num_heads)
        # Absolute value and sum over batch
        contribution = contribution.abs().sum(dim=0)  # (num_heads,)
        importance_accumulator[layer_idx] += contribution

    # Temporarily modify forward functions to register hooks
    original_forwards = [layer.attention.attention.forward for layer in model.vit.encoder.layer]

    def new_forward(self, hidden_states, head_mask=None, output_attentions=False):
        # Standard ViT forward pass up to context_layer
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # Apply dropout using F.dropout with attention_probs_dropout_prob from config
        attention_probs = F.dropout(attention_probs, p=self.config.attention_probs_dropout_prob, training=self.training)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)  # (batch_size, num_heads, seq_len, head_dim)
        # Detach and register hook
        context_layer_detached = context_layer.detach()
        context_layer.register_hook(lambda grad: hook_fn(grad, context_layer_detached, self.layer_idx))

        # Continue original forward pass
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_shape)

        outputs = (context_layer,)
        if output_attentions:
            outputs += (attention_probs,)
        return outputs

    # Apply new forward function to each attention module
    for layer_idx, layer in enumerate(model.vit.encoder.layer):
        layer.attention.attention.forward = new_forward.__get__(layer.attention.attention)
        layer.attention.attention.layer_idx = layer_idx

    # Process batches
    for i, batch in enumerate(tqdm(data_loader, desc="Computing importance scores", total=num_batches)):
        if i >= num_batches:
            break
        images = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs.logits, labels)
        loss.backward()

        total_samples += images.size(0)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Restore original forward functions
    for layer, original_forward in zip(model.vit.encoder.layer, original_forwards):
        layer.attention.attention.forward = original_forward

    # Compute average importance scores
    importance_scores = [accum / total_samples for accum in importance_accumulator]
    # Convert to list of in python lists
    importance_scores = [score.cpu().tolist() for score in importance_scores]
    return importance_scores


# Function to create a model with a specific head removed
def create_pruned_model(original_model, layer_idx_to_prune, head_idx_to_prune):
    # Create a deep copy of the original model
    pruned_model = copy.deepcopy(original_model)
    
    # Get dimensions
    config = original_model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    hidden_size = config.hidden_size
    
    # Get the layer to prune
    layer_to_prune = pruned_model.vit.encoder.layer[layer_idx_to_prune]
    old_attention = layer_to_prune.attention.attention
    
    # Create new attention layer with reduced heads
    new_attention = PrunedViTSelfAttention(config, num_heads - 1).to(device)
    
    # Create head mask (remove the specified head)
    head_mask = torch.ones(num_heads, dtype=torch.bool)
    head_mask[head_idx_to_prune] = False
    
    # Copy weights for query, key, value (excluding the pruned head)
    for module_name in ['query', 'key', 'value']:
        old_module = getattr(old_attention, module_name)
        new_module = getattr(new_attention, module_name)
        
        # Reshape weights to [num_heads, head_dim, hidden_size]
        old_weight = old_module.weight.view(num_heads, head_dim, hidden_size)
        old_bias = old_module.bias.view(num_heads, head_dim) if old_module.bias is not None else None
        
        # Remove the specified head
        new_weight = old_weight[head_mask].reshape(-1, hidden_size)
        new_module.weight.data = new_weight
        
        if old_bias is not None:
            new_bias = old_bias[head_mask].reshape(-1)
            new_module.bias.data = new_bias
    
    # Replace the attention layer
    layer_to_prune.attention.attention = new_attention
    
    # Update the output dense layer to match new input size
    old_dense = layer_to_prune.attention.output.dense
    new_input_size = (num_heads - 1) * head_dim
    new_dense = nn.Linear(new_input_size, hidden_size).to(device)
    
    # Copy weights (removing columns corresponding to pruned head)
    start_idx = head_idx_to_prune * head_dim
    end_idx = (head_idx_to_prune + 1) * head_dim
    
    # Create input mask
    input_mask = torch.ones(hidden_size, dtype=torch.bool)
    input_mask[start_idx:end_idx] = False
    
    # Copy weights excluding pruned head dimensions
    new_dense.weight.data = old_dense.weight[:, input_mask]
    new_dense.bias.data = old_dense.bias.data.clone()
    
    # Replace the dense layer
    layer_to_prune.attention.output.dense = new_dense
    
    return pruned_model

# Training function
def train_model(model, num_epochs=3,train_loader=None, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            try:
                images = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            except Exception as e:
                print(f"Error during training: {e}")
                continue
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# Evaluation function
def evaluate_model(model, test_loader=None):
    accuracy_metric = evaluate.load("accuracy")
    model = model.eval()
    model = nn.DataParallel(model).to(device)
    predictions, labels = [], []
    for inputs in tqdm(test_loader, desc="Evaluating"):
        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = model(pixel_values=inputs["pixel_values"].to(device))
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(inputs["labels"].cpu().numpy())
    model = model.module
    model = model.to('cpu')
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return accuracy["accuracy"]


def main(args):


    # Setup logging
    NOW = str(datetime.datetime.now()).replace(" ", "--")
    log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    #log_dir = f'{args.log_dir}/{NOW}_dataset[{args.dataset}]_trainable[{args.trainable}]_epochs[{args.epochs}]_lr[{args.lr}]_bs[{args.batch_size}]/'
    
    log_dir = f'{args.log_dir}/{NOW}_dataset[{args.dataset.replace("/", "-")}]_trainable[{args.trainable}]_epochs[{args.epochs}]_lr[{args.lr}]_bs[{args.batch_size}]/'

    init_logs(log_file_name, args, log_dir)
    args.logger = get_logger()

    # Authenticate for ImageNet-1k
    if args.dataset == "imagenet-1k" and args.huggingface_token:
        login(token=args.huggingface_token)
        args.logger.info("Authenticated with Hugging Face token for ImageNet-1k")

    # Load dataset
    print("loading dataset...")
    dataset = load_dataset(args.dataset, cache_dir=args.cache_dir,trust_remote_code=True)


    if args.dataset == "cifar100":
        dataset = dataset.rename_column("fine_label", "label")
        train_val = dataset["train"].train_test_split(test_size=0.2, seed=123)
        dataset["train"] = train_val["train"]
        dataset["validation"] = train_val["test"]
    elif args.dataset == "imagenet-1k":
        # ImageNet-1k already has train/validation splits
        if "validation" not in dataset:
            args.logger.warning("No validation split found, splitting train set")
            train_val = dataset["train"].train_test_split(test_size=0.2, seed=123)
            dataset["train"] = train_val["train"]
            dataset["validation"] = train_val["test"]
        if args.subsample == True:
            args.logger.info("Subsampling ImageNet-1k: 5000 training, 1000 validation images")
            train_subset = dataset["train"].train_test_split(train_size=2000, seed=123, stratify_by_column="label")["train"]
            val_subset = dataset["validation"].train_test_split(train_size=1000, seed=123, stratify_by_column="label")["train"]
            dataset["train"] = train_subset
            dataset["validation"] = val_subset
        dataset['train'] = dataset['train'].rename_column("image", "img")
        dataset['validation'] = dataset['validation'].rename_column("image", "img")

        dataset['train'] = dataset['train'].cast_column("img", DatasetImage(decode=False))
        dataset['validation'] = dataset['validation'].cast_column("img", DatasetImage(decode=False))
    elif args.dataset == "cifar10": 
        train_val = dataset["train"].train_test_split(test_size=0.2, seed=123)
        dataset["train"] = train_val["train"]
        dataset["validation"] = train_val["test"]
    elif args.dataset == "slegroux/tiny-imagenet-200-clean":
        dataset['train'] = dataset['train'].rename_column("image", "img")
        dataset['validation'] = dataset['validation'].rename_column("image", "img")
    elif args.dataset == "zh-plus/tiny-imagenet":
        dataset['train'] = dataset['train'].rename_column("image", "img")
        dataset['validation'] = dataset['valid'].rename_column("image", "img")
    else:
        args.logger.warning("Dataset not supported. Please use cifar10, cifar100, imagenet-1k, or slegroux/tiny-imagenet-200-clean.")
        sys.exit(1)




    
    

    labels = dataset["train"].features["label"].names
    args.labels = labels
    print("loading processor...")
    # Initialize processor and dataset
    processor = AutoImageProcessor.from_pretrained(args.processor_name, cache_dir=args.cache_dir, use_fast=True)
    #prepared_ds = dataset.with_transform(functools.partial(transform, processor=processor))

    from PIL import Image
    from io import BytesIO

    def transform(example_batch, processor):
        images = []
        labels = []

        for img_info, label in zip(example_batch["img"], example_batch["label"]):
            try:
                img_bytes = img_info["bytes"]
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Skipping corrupted image: {e}")

        if not images:
            # Return dummy image and label if all failed
            dummy_input = processor([Image.new("RGB", (224, 224))], return_tensors="pt")
            dummy_input["labels"] = torch.tensor([-1])
            return dummy_input

        inputs = processor(images, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs

        

    prepared_ds = {
    "train": dataset["train"].with_transform(functools.partial(transform, processor=processor)),
    "validation": dataset["validation"].with_transform(functools.partial(transform, processor=processor)),
    }
    print("train loader...")
    # Create data loaders
    train_dataloader = DataLoader(
        prepared_ds["train"], batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,persistent_workers=True, drop_last=True
    )
    print("test loader...")
    test_dataloader = DataLoader(
        prepared_ds["validation"], batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,persistent_workers=True, drop_last=False
    )
    print("loading pretrained model...")
    # Initialize model
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True,
        cache_dir=args.cache_dir,
    )


    # Reordering dataset
    # Create a subset and apply the same transform as the main dataset
    reordering_subset = dataset["train"].select(range(0, 10 * args.batch_size, 1)).with_transform(
        functools.partial(transform, processor=processor)
    )
    reorder_dataloader = DataLoader(
        reordering_subset, 
        batch_size=8,  # Match main batch size for consistency
        shuffle=False, 
        num_workers=2, 
        pin_memory=True, 
        persistent_workers=True, 
        drop_last=False
    )
    args.reorder_dataloader = reorder_dataloader




    if args.model_name == "google/vit-base-patch16-224":

        # Define elastic config for NAS
        regular_config = {
            "atten_out_space": [768],
            "inter_hidden_space": [3072],
            "residual_hidden": [768],
        }
        elastic_config = {
            "atten_out_space": [768],
            # work on this
            "inter_hidden_space": [2304, 1536, 1020, 768],
            "residual_hidden": [768],
        }
        config = {
            str(i): elastic_config if i in [1, 2, 3, 4, 5, 6,7,8, 9,10, 11] else regular_config for i in range(12)
        }
        config["layer_elastic"] = {
            "elastic_layer_idx": [1, 2, 4,5,7,8, 9,10],
            "remove_layer_prob": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        }
    elif args.model_name == "facebook/deit-tiny-patch16-224":
        # Define elastic config for NAS
        regular_config = {
            "atten_out_space": [192],
            "inter_hidden_space": [768],
            "residual_hidden": [192],
        }
        elastic_config = {
            "atten_out_space": [192],
            # work on this
            "inter_hidden_space": [192, 288, 384, 576, 768],
            "residual_hidden": [192],
        }
        config = {
            str(i): elastic_config if i in [1,2, 3,4,5,6,7,8,9,10,11] else regular_config for i in range(12)
        }
        config["layer_elastic"] = {
            "elastic_layer_idx": [2,3,6,7,8],
            "remove_layer_prob": [.5, 0.5, 0.5, 0.5, 0.5]
        }
    elif args.model_name == "facebook/deit-small-patch16-224":
        # Define elastic config for NAS
        regular_config = {
            "atten_out_space": [384],
            "inter_hidden_space": [1536],
            "residual_hidden": [384],
        }
        elastic_config = {
            "atten_out_space": [384],
            # work on this
            "inter_hidden_space": [384, 576, 768, 1024, 1536],
            "residual_hidden": [384],
        }
        config = {
            str(i): elastic_config if i in [1,2, 3,4,5,6,7,8,9,10,11] else regular_config for i in range(12)
        }
        config["layer_elastic"] = {
            "elastic_layer_idx": [2,3,6,7,8],
            "remove_layer_prob": [.5, 0.5, 0.5, 0.5, 0.5]
        }


    # logging.info("Training ViT on %s dataset with model %s", args.dataset, args.model_name)
    # original_accuracy = evaluate_model(model, test_loader=test_dataloader)
    # logging.info(
    # f"Original Model - Accuracy: {original_accuracy:.2f}% ")
    # logging.info("Model training ...")
    # # Train the model
    # train_model(model, num_epochs=2, train_loader=train_dataloader, device=device)






    # #Make sure the save directory exists
    # args.save_path = "./bestmodel"
    # os.makedirs(args.save_path, exist_ok=True)

    # best_accuracy = evaluate_model(model, test_loader=test_dataloader)
    # logging.info(f"Initial accuracy: {best_accuracy:.4f}")

    # for iteration in range(200):
    #     print(f"\n=== Iteration {iteration + 1}/200 ===")
    #     logging.info(f"Iteration {iteration + 1}/200")
        
    #     # Train model for 2 epochs
    #     train_model(model, num_epochs=1, train_loader=train_dataloader, device=device)
        
    #     # Evaluate the model
    #     current_accuracy = evaluate_model(model, test_loader=test_dataloader)
    #     print(f"Accuracy after iteration {iteration + 1}: {current_accuracy:.4f}")
    #     logging.info(f"Iteration {iteration + 1} - Accuracy: {current_accuracy:.4f}")
        
    #     # Save if accuracy improved
    #     if current_accuracy > best_accuracy:
    #         best_accuracy = current_accuracy
    #         save_path = os.path.join(args.save_path, f"best_model_iter{iteration + 1}.pt")
    #         # add iteration number to save path and save model
    #         model.save_pretrained(save_path)
    #         logging.info(f"New best model saved to {save_path} with accuracy {best_accuracy:.4f}")

    #         logging.info(f"Model saved to {save_path} with accuracy {best_accuracy:.4f}")
    #     else:
    #         logging.info("No improvement in accuracy, model not saved.")
    

    # Evaluate original model
    original_accuracy = evaluate_model(model, test_loader=test_dataloader)
    original_params = calculate_params(model)
    original_flops = compute_flops(model)
    logging.info(
    f"FineTuned Model - Accuracy: {original_accuracy:.2f}%, "
    f"Params: {original_params:.2f}M, FLOPs: {original_flops[1]:.2f}G")
    # Calculate scores
    logging.info("Calculating head scores...")
    scores1 = get_head_scores_QKV(model)
    scores2 = get_head_scores_Output(model)
    scores3 = get_head_scores_OutputActivation(model, reorder_dataloader, device, max_samples=1000)
    scores4 = get_head_scores_Paper(model, reorder_dataloader, device, num_batches=10)
    # implement adaptive pruning based on scores4
    # remove the heads with the lowest score
    # then recompute scorees4




    
    # for layer_idx and head_idx in range 0-12
    logging.info("Testing pruned models...")
    for layer_idx in range(12):
        for head_idx  in range(12):
            if layer_idx >= len(scores4) or head_idx >= len(scores4[layer_idx]):
                continue
            print(f"\nTesting pruned model: Layer {layer_idx}, Head {head_idx}")
            try:
                pruned_model = create_pruned_model(model, layer_idx, head_idx)
                accuracy = evaluate_model(pruned_model, test_loader=test_dataloader)
                params = calculate_params(pruned_model)
                flops = compute_flops(pruned_model)
                logging.info(f"Layer {layer_idx}, Head {head_idx}, Score1: {scores1[layer_idx][head_idx]:.4f},Score2: {scores2[layer_idx][head_idx]:.4f},Score3: {scores3[layer_idx][head_idx]:.4f}, Score4: {scores4[layer_idx][head_idx]:.4f}, "f"Accuracy: {accuracy:.2f}%, Params: {params:.2f}M, FLOPs: {flops[1]:.2f}G")
                
                
                # Clean up
                del pruned_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"Error with Layer {layer_idx}, Head {head_idx}: {str(e)}")

    



if __name__ == "__main__":
    args = arguments()
    main(args)
