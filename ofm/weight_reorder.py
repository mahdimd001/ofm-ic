import torch
import torch.nn as nn
import torch.nn.init as init
from functools import partial
import copy
from collections import defaultdict

def mlp_masking(model, sparcity=.5, method='magnitude'):
    sam_vit_layers = model.vision_encoder.layers

    # Ensure sparsity is between 0 and 100
    assert 0 <= sparcity <= 1, "Sparcity should be a value between 0 and 1."

    # Convert the percentage to a fraction
    fraction = sparcity

    for i, layer in enumerate(sam_vit_layers):
        W1 = layer.mlp.lin1.weight  # shape: (3072, 768)
        W2 = layer.mlp.lin2.weight  # shape: (768, 3072)
        b1 = layer.mlp.lin1.bias  # bias of lin1

        # Flatten weights to easily sort and prune
        W1_flat = W1.view(-1)
        W2_flat = W2.view(-1)

        # Determine the number of elements to mask
        num_to_mask_W1 = int(fraction * W1_flat.numel())
        num_to_mask_W2 = int(fraction * W2_flat.numel())

        if method == 'magnitude':
            # Sort W1 and W2 by absolute magnitude and get the threshold values
            threshold_W1 = torch.topk(W1_flat.abs(), num_to_mask_W1, largest=False).values.max()
            threshold_W2 = torch.topk(W2_flat.abs(), num_to_mask_W2, largest=False).values.max()

            # Mask out the parameters below the threshold by setting them to zero
            W1_mask = W1.abs() >= threshold_W1
            W2_mask = W2.abs() >= threshold_W2
        
        else:
            # Generate random indices for W1 and W2 to mask
            W1_indices_to_mask = torch.randperm(W1_flat.numel())[:num_to_mask_W1]
            W2_indices_to_mask = torch.randperm(W2_flat.numel())[:num_to_mask_W2]

            # Create masks initialized to all ones (keep all)
            W1_mask = torch.ones_like(W1_flat)
            W2_mask = torch.ones_like(W2_flat)

            # Set the random indices to zero (mask them)
            W1_mask[W1_indices_to_mask] = 0
            W2_mask[W2_indices_to_mask] = 0

            # Reshape the masks back to original dimensions of W1 and W2
            W1_mask = W1_mask.view(W1.shape)
            W2_mask = W2_mask.view(W2.shape)

        # Apply the mask to W1 and W2
        W1.data.mul_(W1_mask)
        W2.data.mul_(W2_mask)

        # Optionally: You could also mask the biases similarly if needed

        print(f"Layer {i}: Masked {num_to_mask_W1} params in W1 and {num_to_mask_W2} params in W2.")

    return model



wanda_sums = {i:[[],[]] for i in range(12)}

# Assuming encoder is already a deep copy of model.vision_encoder
def randomize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            init.kaiming_normal_(layer.weight)  # You can use other initializations too
            if layer.bias is not None:
                init.zeros_(layer.bias)


def mlp_forward_hook(inst, inp, out, layer, lin):
    W = inst.weight  # shape: (3072, 768)

    #print(f'inst : {inst} \t layer : {layer} \t lin : {lin}')
    #print(f'\tW : {W.shape}')

    C_out = W.shape[1]
    l2_norm = inp[0].view(-1,C_out)
    l2_norm = l2_norm.norm(p=2, dim=0)

    #print(f'\tl2_norm : {l2_norm.shape}')

    wanda = W.abs() * l2_norm

    if lin == 1:
        row_sums = torch.abs(wanda).sum(dim=1)
        wanda_sums[layer][0].append(row_sums)
    
    elif lin == 2:
        column_sums = torch.abs(wanda).sum(dim=0)
        wanda_sums[layer][1].append(column_sums)
    
    #print(f'\twanda : {wanda.shape}')

    #return wanda

def movement_reordering(model, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    grads = {i:[[],[]] for i in range(12)}

    loss_func = nn.MSELoss()

    encoder = copy.deepcopy(model.vision_encoder).to(device)

    # Randomize the weights of the encoder for non-zero grads
    randomize_weights(encoder)

    encoder.train()
    for idx,(inputs, labels) in enumerate(dataloader):
        data = {'pixel_values': torch.stack([d['pixel_values'].squeeze(0) for d in inputs])}
        print(f'data["pixel_values"] : {data["pixel_values"].shape}')
        output = encoder(data["pixel_values"].to(device))

        pred_embeddings = output[0]
        gts_embeddings = torch.stack(labels).to(device)
        loss = loss_func(gts_embeddings, pred_embeddings)

        loss.backward()

        #Capture grads for lin1 and lin2
        for idx, layer in enumerate(encoder.layers):
            G1 = layer.mlp.lin1.weight.grad
            G2 = layer.mlp.lin2.weight.grad
            row_sums = G1.abs().sum(dim=1) #G1.abs()
            column_sums = G2.abs().sum(dim=0) #G2.abs()
            print(f'Layer : {idx}')
            print("\tlin1 grads:", G1.shape)
            print("\tlin2 grads:", G2.shape)
            grads[idx][0].append(row_sums)
            grads[idx][1].append(column_sums)
        
        # # Zero out gradients
        # for param in encoder.parameters():
        #     if param.grad is not None:
        #         param.grad.zero_()
    
    score_dist = {}
    print(f'Aggregating movement sums')
    for (k,v),layer in zip(grads.items(),encoder.layers):
        grad_row_sums = sum(v[0]) / len(v[0])
        grad_column_sums = sum(v[1]) / len(v[1])

        W1 = layer.mlp.lin1.weight  # shape: (3072, 768)
        W2 = layer.mlp.lin2.weight  # shape: (768, 3072)
        b1 = layer.mlp.lin1.bias

        weight_row_sums = W1.abs().sum(dim=1) #W1.abs() 
        weight_column_sums = W2.abs().sum(dim=0) #W2.abs()
        
        avg_row_sums = grad_row_sums.abs() * weight_row_sums
        avg_column_sums = grad_column_sums.abs() * weight_column_sums

        avg_sums = (avg_row_sums + avg_column_sums) / 2

        score_dist[k] = avg_sums.flatten().tolist()

        _, sorted_indices = avg_sums.sort(descending=True)

        print(f'{k} --> {avg_sums.shape}')
        # print(f'\tgrad_row_sums : {grad_row_sums}')
        # print(f'\tweight_row_sums : {weight_row_sums}')
        # print(f'\tavg_row_sums : {avg_row_sums}')
        # print(f'\tavg_sums : {avg_sums}')

        W1_sorted = W1[sorted_indices, :] #sort rows of W1
        W2_sorted = W2[ :, sorted_indices ] #sort columns of W2
        b1_sorted = b1[sorted_indices] #sort b1


        #Re-order weights and bias of lin1 and lin2
        layer.mlp.lin1.weight.data = W1_sorted
        layer.mlp.lin2.weight.data = W2_sorted
        layer.mlp.lin1.bias.data = b1_sorted
    
    return score_dist

def wanda_reordering(model,dataloader):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = model.vision_encoder.to(device)

    hooks_1, hooks_2 = [],[]


    for idx, layer in enumerate(encoder.layers):
        hook_1 = layer.mlp.lin1.register_forward_hook(partial(mlp_forward_hook, layer=idx, lin=1)) #module.register_backward_hook)
        hook_2 = layer.mlp.lin2.register_forward_hook(partial(mlp_forward_hook, layer=idx, lin=2))
        hooks_1.append(hook_1)
        hooks_2.append(hook_2)
    
    #encoder = nn.DataParallel(encoder)
    encoder.eval()

    with torch.no_grad():
        for idx,(inputs) in enumerate(dataloader):
            #data = {'pixel_values': torch.stack([d['pixel_values'].squeeze(0) for d in inputs])}
            #print(f'data["pixel_values"] : {data["pixel_values"].shape}')
            output = encoder(inputs["pixel_values"].to(device))
    
        for hook_1,hook_2 in zip(hooks_1,hooks_2):
            hook_1.remove()
            hook_2.remove()

    score_dist = {}
    #print(f'Aggregating wanda sums')
    for (k,v),layer in zip(wanda_sums.items(),encoder.layers):
        avg_sums = ((sum(v[0]) / len(v[0])) + (sum(v[1]) / len(v[1]))) / 2
        #print(f'{k} --> {avg_sums.shape}')

        score_dist[k] = avg_sums.flatten().tolist()

        _, sorted_indices = avg_sums.sort(descending=True)

        W1 = layer.mlp.lin1.weight  # shape: (3072, 768)
        W2 = layer.mlp.lin2.weight  # shape: (768, 3072)
        b1 = layer.mlp.lin1.bias

        W1_sorted = W1[sorted_indices, :] #sort rows of W1
        W2_sorted = W2[ :, sorted_indices ] #sort columns of W2
        b1_sorted = b1[sorted_indices] #sort b1


        #Re-order weights and bias of lin1 and lin2
        layer.mlp.lin1.weight.data = W1_sorted
        layer.mlp.lin2.weight.data = W2_sorted
        layer.mlp.lin1.bias.data = b1_sorted
    
    return score_dist


def magnitude_reordering(sam_vit_layers):

    score_dist = {}
    
    for i, layer in enumerate(sam_vit_layers):

        W1 = layer.mlp.lin1.weight  # shape: (3072, 768)
        W2 = layer.mlp.lin2.weight  # shape: (768, 3072)
        b1 = layer.mlp.lin1.bias
        
        row_sums = W1.sum(dim=1)
        column_sums = W2.sum(dim=0)
        avg_sums = (row_sums + column_sums) / 2
        score_dist[i] = avg_sums.flatten().tolist()

        _, sorted_indices = avg_sums.sort(descending=True) #descending=True

        W1_sorted = W1[sorted_indices, :] #sort rows of W1
        W2_sorted = W2[ :, sorted_indices ] #sort columns of W2
        b1_sorted = b1[sorted_indices] #sort b1


        #Re-order weights and bias of lin1 and lin2
        layer.mlp.lin1.weight.data = W1_sorted
        layer.mlp.lin2.weight.data = W2_sorted
        layer.mlp.lin1.bias.data = b1_sorted
    
    return score_dist


def mask_layers(model, layer_indices_to_mask):
    """
    Masks specified layers in the model by setting their parameters to zero.

    Args:
        model (torch.nn.Module): The model containing the layers.
        layer_indices_to_mask (list): List of layer indices to mask.

    Returns:
        torch.nn.Module: The modified model with masked layers.
    """
    for idx, layer in enumerate(model.vision_encoder.layers):
        if idx in layer_indices_to_mask:
            # Zero out the parameters in the attention sub-layer
            layer.attn.qkv.weight.data.zero_()
            layer.attn.qkv.bias.data.zero_()
            layer.attn.proj.weight.data.zero_()
            layer.attn.proj.bias.data.zero_()

            # Zero out the parameters in the MLP sub-layer
            layer.mlp.lin1.weight.data.zero_()
            layer.mlp.lin1.bias.data.zero_()
            layer.mlp.lin2.weight.data.zero_()
            layer.mlp.lin2.bias.data.zero_()

            # Zero out the LayerNorm parameters if desired (optional)
            layer.layer_norm1.weight.data.zero_()
            layer.layer_norm1.bias.data.zero_()
            layer.layer_norm2.weight.data.zero_()
            layer.layer_norm2.bias.data.zero_()

    return model

def remove_layers(model, layer_indices_to_remove):
    """
    Removes specified layers from the model by their indices.

    Args:
        model (torch.nn.Module): The model containing the layers.
        layer_indices_to_remove (list): List of layer indices to remove.

    Returns:
        torch.nn.Module: The modified model with specified layers removed.
    """
    # Sort the indices in descending order to avoid index shifting issues
    layer_indices_to_remove = sorted(layer_indices_to_remove, reverse=True)
    
    # Iterate over the indices and remove the corresponding layers
    for idx in layer_indices_to_remove:
        del model.vision_encoder.layers[idx]
    
    return model


def sam_weight_reorder(model, dataloader=None, method='magnitude'):
    """_summary_

    Args:
        model (torch.module): Pytorch model
        order (int, optional): Order used to compute importance. Defaults to 0.

    Returns:
        torch.module: Model
    """

    if method == 'wanda':
        score_dist = wanda_reordering(model, dataloader)

    elif method == 'magnitude':
        #model = model.to('cpu')
        sam_vit_layers = model.vision_encoder.layers
        score_dist = magnitude_reordering(sam_vit_layers)
    elif method == 'movement':
        score_dist = movement_reordering(model,dataloader)
        
    return model, score_dist













# work fine (samani)
def vit_magnitude_reordering(vit_layers):
    """Compute magnitude-based importance scores for ViT MLP blocks."""
    score_dist = {}
    for i, layer in enumerate(vit_layers):
        W1 = layer.intermediate.dense.weight  # [3072, 768]
        W2 = layer.output.dense.weight  # [768, 3072]
        b1 = layer.intermediate.dense.bias
        row_sums = W1.abs().sum(dim=1)
        column_sums = W2.abs().sum(dim=0)
        avg_sums = (row_sums + column_sums) / 2
        score_dist[i] = avg_sums.flatten().tolist()
        _, sorted_indices = avg_sums.sort(descending=True)
        W1_sorted = W1[sorted_indices, :]
        W2_sorted = W2[:, sorted_indices]
        b1_sorted = b1[sorted_indices]
        layer.intermediate.dense.weight.data = W1_sorted
        layer.output.dense.weight.data = W2_sorted
        layer.intermediate.dense.bias.data = b1_sorted
    return score_dist

# work fine (samani)
def vit_wanda_reordering(model, dataloader):
    """Compute Wanda-based importance scores for ViT MLP blocks."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    global wanda_sums
    wanda_sums = {i: [[], []] for i in range(len(model.vit.encoder.layer))}
    hooks_1, hooks_2 = [], []
    for idx, layer in enumerate(model.vit.encoder.layer):
        hook_1 = layer.intermediate.dense.register_forward_hook(partial(mlp_forward_hook, layer=idx, lin=1))
        hook_2 = layer.output.dense.register_forward_hook(partial(mlp_forward_hook, layer=idx, lin=2))
        hooks_1.append(hook_1)
        hooks_2.append(hook_2)
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["pixel_values"].to(device)
            model(inputs)
    for hook_1, hook_2 in zip(hooks_1, hooks_2):
        hook_1.remove()
        hook_2.remove()
    score_dist = []
    for idx, layer in enumerate(model.vit.encoder.layer):
        avg_sums = ((sum(wanda_sums[idx][0]) / len(wanda_sums[idx][0])) + (sum(wanda_sums[idx][1]) / len(wanda_sums[idx][1]))) / 2
        score_dist.append(avg_sums)
        _, sorted_indices = avg_sums.sort(descending=True)
        W1 = layer.intermediate.dense.weight
        W2 = layer.output.dense.weight
        b1 = layer.intermediate.dense.bias
        W1_sorted = W1[sorted_indices, :]
        W2_sorted = W2[:, sorted_indices]
        b1_sorted = b1[sorted_indices]
        layer.intermediate.dense.weight.data = W1_sorted
        layer.output.dense.weight.data = W2_sorted
        layer.intermediate.dense.bias.data = b1_sorted
    return score_dist

def vit_movement_reordering(model, dataloader):
    """Compute movement-based importance scores for ViT MLP blocks."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = copy.deepcopy(model.vit.encoder).to(device)
    encoder.train()
    loss_func = nn.CrossEntropyLoss()
    grads = {i: [[], []] for i in range(len(encoder.layer))}
    randomize_weights(encoder)
    for batch in dataloader:
        inputs = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        encoder.zero_grad()
        outputs = encoder(inputs)[0]
        head = nn.Linear(768, 10).to(device)
        logits = head(outputs[:, 0, :])
        loss = loss_func(logits, labels)
        loss.backward()
        for idx, layer in enumerate(encoder.layer):
            G1 = layer.intermediate.dense.weight.grad
            G2 = layer.output.dense.weight.grad
            if G1 is not None and G2 is not None:
                row_sums = G1.abs().sum(dim=1)
                column_sums = G2.abs().sum(dim=0)
                grads[idx][0].append(row_sums)
                grads[idx][1].append(column_sums)
    score_dist = []
    for idx, layer in enumerate(encoder.layer):
        grad_row_sums = sum(grads[idx][0]) / len(grads[idx][0]) if grads[idx][0] else torch.zeros(3072, device=device)
        grad_column_sums = sum(grads[idx][1]) / len(grads[idx][1]) if grads[idx][1] else torch.zeros(3072, device=device)
        W1 = layer.intermediate.dense.weight
        W2 = layer.output.dense.weight
        b1 = layer.intermediate.dense.bias
        weight_row_sums = W1.abs().sum(dim=1)
        weight_column_sums = W2.abs().sum(dim=0)
        avg_row_sums = grad_row_sums.abs() * weight_row_sums
        avg_column_sums = grad_column_sums.abs() * weight_column_sums
        avg_sums = (avg_row_sums + avg_column_sums) / 2
        score_dist.append(avg_sums)
        _, sorted_indices = avg_sums.sort(descending=True)
        W1_sorted = W1[sorted_indices, :]
        W2_sorted = W2[:, sorted_indices]
        b1_sorted = b1[sorted_indices]
        layer.intermediate.dense.weight.data = W1_sorted
        layer.output.dense.weight.data = W2_sorted
        layer.intermediate.dense.bias.data = b1_sorted
    return score_dist

#samani for vit support
def vit_weight_reorder(model, dataloader=None, method='magnitude'):
    """
    Reorder weights in ViT's MLP blocks using specified method.

    Args:
        model (nn.Module): ViT model (e.g., ViTForImageClassification).
        dataloader (DataLoader, optional): DataLoader for data-dependent methods like Wanda.
        method (str): Reordering method ('magnitude', 'wanda', 'movement'). Defaults to 'magnitude'.

    Returns:
        tuple: (model, score_dist)
            - model: Reordered ViT model.
            - score_dist: List of importance scores for each MLP block's intermediate dimension.
    """
    if method == 'wanda':
        score_dist = vit_wanda_reordering(model, dataloader)
    elif method == 'magnitude':
        vit_layers = model.vit.encoder.layer
        score_dist = vit_magnitude_reordering(vit_layers)
    elif method == 'movement':
        score_dist = vit_movement_reordering(model, dataloader)
    else:
        raise ValueError(f"Unsupported reordering method: {method}")

    return model, score_dist




def compute_global_ffn_allocation(model, target_param,compute_score = 'magnitude',dataloader=None):
    """
    Use Mahdi's score formula to compute global FFN neuron allocation under a total parameter budget.
    Reorders weights in-place per layer and returns how many FFN units to keep per layer.
    """
    all_units = [] 
    if compute_score == 'magnitude':
        for i, layer in enumerate(model.vit.encoder.layer):
            W1 = layer.intermediate.dense.weight     # [out_dim, in_dim]
            W2 = layer.output.dense.weight           # [in_dim, out_dim]
            b1 = layer.intermediate.dense.bias

            row_sums = W1.abs().sum(dim=1)           # [out_dim]
            column_sums = W2.abs().sum(dim=0)        # [out_dim]
            avg_sums = (row_sums + column_sums) / 2  # [out_dim]

            avg_sums = avg_sums.cpu()
            sorted_scores, sorted_indices = avg_sums.sort(descending=True)

            # Reorder weights in-place (as you already do)
            # W1_sorted = W1[sorted_indices, :]
            # W2_sorted = W2[:, sorted_indices]
            # b1_sorted = b1[sorted_indices]

            # layer.intermediate.dense.weight.data = W1_sorted
            # layer.output.dense.weight.data = W2_sorted
            # layer.intermediate.dense.bias.data = b1_sorted
            if i != 0: # Ensure the first layer keeps all units
                # normalize scores to be in the range [0, 1]
                #sorted_scores = (sorted_scores - sorted_scores.min()) / (sorted_scores.max() - sorted_scores.min()+1e-8)
                for j, score in enumerate(sorted_scores):
                    
                    all_units.append((score.item(), i, j))
        target_param = target_param -768 # Subtract the first layer's units (768) from the target_param
        # Sort all units across all layers globally
        all_units.sort(key=lambda x: x[0], reverse=True)

        all_units = all_units[:target_param]  # Keep only the top N units based on target_param
        keep_per_layer = defaultdict(int)
        for tup in all_units:
            second_value = tup[1]
            keep_per_layer[second_value] += 1
        keep_per_layer[0] = 768  # Ensure the first layer keeps all units
        return keep_per_layer
    elif compute_score == 'wanda':
        # Implement Wanda-based global FFN allocation here
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        global wanda_sums
        wanda_sums = {i: [[], []] for i in range(len(model.vit.encoder.layer))}
        hooks_1, hooks_2 = [], []
        for idx, layer in enumerate(model.vit.encoder.layer):
            hook_1 = layer.intermediate.dense.register_forward_hook(partial(mlp_forward_hook, layer=idx, lin=1))
            hook_2 = layer.output.dense.register_forward_hook(partial(mlp_forward_hook, layer=idx, lin=2))
            hooks_1.append(hook_1)
            hooks_2.append(hook_2)
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["pixel_values"].to(device)
                model(inputs)
        for hook_1, hook_2 in zip(hooks_1, hooks_2):
            hook_1.remove()
            hook_2.remove()
        score_dist = []
        for idx, layer in enumerate(model.vit.encoder.layer):
            avg_sums = ((sum(wanda_sums[idx][0]) / len(wanda_sums[idx][0])) + (sum(wanda_sums[idx][1]) / len(wanda_sums[idx][1]))) / 2
            score_dist.append(avg_sums)
            sorted_scores, sorted_indices = avg_sums.sort(descending=True)
            # W1 = layer.intermediate.dense.weight
            # W2 = layer.output.dense.weight
            # b1 = layer.intermediate.dense.bias
            # W1_sorted = W1[sorted_indices, :]
            # W2_sorted = W2[:, sorted_indices]
            # b1_sorted = b1[sorted_indices]
            # layer.intermediate.dense.weight.data = W1_sorted
            # layer.output.dense.weight.data = W2_sorted
            # layer.intermediate.dense.bias.data = b1_sorted
            # Save individual unit scores and costs
            if idx != 0:  # Ensure the first layer keeps all units
                for j, score in enumerate(sorted_scores):
                    
                    all_units.append((score.item(), idx, j))
        all_units.sort(key=lambda x: x[0], reverse=True)
        target_param = target_param - 768
        all_units = all_units[:target_param]  # Keep only the top N units based on target_param
        keep_per_layer = defaultdict(int)
        for tup in all_units:
            second_value = tup[1]
            keep_per_layer[second_value] += 1
        keep_per_layer[0] = 768
        return keep_per_layer


def global_vit_magnitude_reordering(vit_layers,NumberOfParams=4500):
    """Compute magnitude-based importance scores for ViT MLP blocks in all layers and then ."""
    return 0

    

def global_vit_wanda_reordering(model, dataloader,NumberOfParams=4500):
    return 0


def vit_global_weight_reorder(model, dataloader=None, method='magnitude',NumberOfParams=4500):
    """
    Reorder weights in ViT's MLP blocks using specified method.

    Args:
        model (nn.Module): ViT model (e.g., ViTForImageClassification).
        dataloader (DataLoader, optional): DataLoader for data-dependent methods like Wanda.
        method (str): Reordering method ('magnitude', 'wanda', 'movement'). Defaults to 'magnitude'.

    Returns:
        tuple: (model, score_dist)
            - model: Reordered ViT model.
            - score_dist: List of importance scores for each MLP block's intermediate dimension.
    """
    if method == 'wanda':
        score_dist = global_vit_wanda_reordering(model, dataloader,NumberOfParams=4500)
    elif method == 'magnitude':
        vit_layers = model.vit.encoder.layer
        score_dist = global_vit_magnitude_reordering(vit_layers,NumberOfParams=4500)
    else:
        raise ValueError(f"Unsupported reordering method: {method}")

    return model, score_dist