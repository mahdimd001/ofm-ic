""" Official Implementation
One Foundation Model Fits All: Single-stage Foundation Model Training with Zero-shot Deployment
"""

import copy
import os
from typing import Any
import torch
from torch import nn
from .model_downsize import (
    bert_module_handler,
    arc_config_sampler,
    vit_module_handler,
    sam_module_handler,
    t5_module_handler,
    roberta_module_handler,
    distilbert_module_handler,
    swin_module_handler,
    mamba_module_handler,
    clip_module_handler,
    test_function
)
from .weight_reorder import sam_weight_reorder, mask_layers, remove_layers, mlp_masking, vit_weight_reorder
#from .prune import prune_magnitude, prune_random
from .param_prioritization import *
from .utils import calculate_params, save_dict_to_file, load_dict_from_file


class OFM:
    def __init__(self, model, elastic_config=None) -> None:
        test_function()
        self.model = model
        self.total_params = calculate_params(model=model)

        if hasattr(self.model.config, "elastic_config"):
            elastic_config = self.model.config.elastic_config

        if not elastic_config:
            # set defalt search space configuration (this is defalt setting for bert)
            elastic_config = {
                0:  {
                        "atten_out_space": [768],
                        "inter_hidden_space": [3072, 1920, 1280],
                        "residual_hidden_space": [768],
                    }
                }
            print(
                f"[Warning]: No elastic configuration provides. Set to the defalt elastic space {elastic_config}."
            )
        elif isinstance(elastic_config, str):
            elastic_config = load_dict_from_file(elastic_config)

        assert isinstance(
            elastic_config, dict
        ), "Invalid elastic_config, expect input a dictionary or file path"

        #elastic_config = {int(k):v for k,v in elastic_config.items()}
        self.model.config.elastic_config = elastic_config
        # self.elastic_config = elastic_config
        self.local_grads = []
        self.alphas = []
        self._pre_global_grad = None
        
    
    @staticmethod
    def check():
        return 'from OFM_SAM package'


    # samani change for vit support
    def random_resource_aware_model(self):
        """_summary_

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """

        if "sam" == self.model.config.model_type.lower():
            arc_config = arc_config_sampler(
                self.model.config.elastic_config,
                n_layer=self.model.vision_encoder.config.num_hidden_layers,
            )
        else:
            arc_config = arc_config_sampler(
                self.model.config.elastic_config,
                n_layer=self.model.config.num_hidden_layers,
            )
        subnetwork, total_params = self.resource_aware_model(arc_config)

        return subnetwork, total_params, arc_config




        # samani change for vit support
    def smart_resource_aware_model(self,layer_size,removed_layers=None):
        """_summary_

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """

        if "vit" == self.model.config.model_type.lower():
            arc_config = arc_config_sampler(
                self.model.config.elastic_config,
                n_layer=self.model.config.num_hidden_layers,
            )
        for i in layer_size:
            arc_config[str(i)]['inter_hidden'] = layer_size[i]
        arc_config['remove_layer_idx'] = removed_layers
        subnetwork, total_params = self.resource_aware_model(arc_config)

        return subnetwork, total_params, arc_config


    # samani change for vit support
    def smallest_model(self):
        """Return the smallest model in the elastic space

        Returns:
            - subnetwork (nn.Module): The smallest model in the elastic space
            - params (int): The number of parameters in million of the smallest model
            - arc_config (dict): The configuration of the smallest model
        """
        print("Getting the smallest model in the elastic space...")
        try:
            if "sam" == self.model.config.model_type.lower():
                arc_config = arc_config_sampler(
                    self.model.config.elastic_config,
                    smallest=True,
                    n_layer=self.model.vision_encoder.config.num_hidden_layers,
                )
            else:
                arc_config = arc_config_sampler(
                    self.model.config.elastic_config,
                    smallest=True,
                    n_layer=self.model.config.num_hidden_layers,
                )
            
            subnetwork, params = self.resource_aware_model(arc_config)
            return subnetwork, params, arc_config
        except Exception as e:
            print(f"Error in getting smallest model: {e}")

    def largest_model(self):
        return copy.deepcopy(self.model), self.total_params, {}
    
    #samani change for vit support
    def mlp_layer_reordering(self,dataloader=None,method='magnitude'):
        if "sam" == self.model.config.model_type.lower():
            self.model, score_dist = sam_weight_reorder(self.model,dataloader,method)
            return score_dist
        elif "vit" == self.model.config.model_type.lower():
            self.model, score_dist = vit_weight_reorder(self.model,dataloader,method)
            return score_dist
        else:
            raise NotImplemented(f'Weight reordering not yet implemented for \
                                 {self.model.config.model_type.lower()}')
    
    def mlp_layer_masking(self,dataloader=None,sparsity=.5,method='magnitude',prune_n=0, prune_m=0):
        if "sam" == self.model.config.model_type.lower():
            if method == 'naive':
                #prune_random(self.model,sparsity)
                prune_magnitude(self.model, sparsity, reverse=True)
            elif method == 'magnitude':
                prune_magnitude(self.model, sparsity, prune_n=prune_n, prune_m=prune_m)
        else:
            raise NotImplemented(f'MLP masking not yet implemented for \
                                 {self.model.config.model_type.lower()}')
    
    def mask_layers(self,layers):
        if "sam" == self.model.config.model_type.lower():
            mask_layers(self.model, layers)
        else:
            raise NotImplemented(f'Masking not yet implemented for \
                                 {self.model.config.model_type.lower()}')
    
    def remove_layers(self,layers):
        if "sam" == self.model.config.model_type.lower():
            remove_layers(self.model, layers)
        else:
            raise NotImplemented(f'Pruning not yet implemented for \
                                 {self.model.config.model_type.lower()}')

    def resource_aware_model(self, arc_config):
        if "bert" == self.model.config.model_type.lower():
            return bert_module_handler(self.model, arc_config)
        elif "vit" == self.model.config.model_type.lower():
            return vit_module_handler(self.model, arc_config)
        elif "sam" == self.model.config.model_type.lower():
            return sam_module_handler(self.model, arc_config)
        elif "t5" == self.model.config.model_type.lower():
            return t5_module_handler(self.model, arc_config)
        elif "roberta" == self.model.config.model_type.lower():
            return roberta_module_handler(self.model, arc_config)
        elif "distilbert" == self.model.config.model_type.lower():
            return distilbert_module_handler(self.model, arc_config)
        elif "swin" == self.model.config.model_type.lower():
            return swin_module_handler(self.model, arc_config)
        elif "mamba" == self.model.config.model_type.lower():
            return mamba_module_handler(self.model, arc_config)
        elif "clip" == self.model.config.model_type.lower():
            return clip_module_handler(self.model, arc_config)
        else:
            raise NotImplementedError

    def salient_parameter_prioritization(self, metric=l1_norm):
        self.model = salient_parameter_prioritization(self.model, metric)

    def grad_accumulate(self, local_grad, alpha=None):
        self.local_grads.append(local_grad)
        self.alphas.append(alpha)

    def apply_grad(self, grad, removed_layer_idx=None):
        """Apply the gradients to the full-size model, adjusting for removed layers.

        Args:
            grad (dict): Trained downsized model gradients.
            removed_layer_idx (list of int, optional): List of layer indices that were removed 
                                                    in the downsized model. Defaults to None.
        """
        if removed_layer_idx is None:
            removed_layer_idx = []

        self.model.to("cpu")
        with torch.no_grad():
            for name, param in self.model.named_parameters():

                # Determine the original layer index
                if 'vision_encoder.layers' in name:
                    layer_idx = int(name.split('.')[2])
                else:
                    layer_idx = None
                    # continue
                

                if layer_idx:
                    if layer_idx in removed_layer_idx:
                        continue
                
                    # Skip the removed layers
                    adjusted_layer_idx = layer_idx - sum(1 for removed_idx in removed_layer_idx if removed_idx < layer_idx)



                    # Replace the layer index in the name for fetching the correct gradient
                    grad_name = name.replace(f'.{layer_idx}.', f'.{adjusted_layer_idx}.')
                else:
                    grad_name = name
                        
                if grad_name in grad:
                    local_grad = grad[grad_name].cpu()

                    slices = tuple(
                        slice(0, min(sm_dim, lg_dim))
                        for sm_dim, lg_dim in zip(local_grad.shape, param.shape)
                    )

                    if self._pre_global_grad:
                        param[slices] -= (
                            0.9 * local_grad + 0.1 * self._pre_global_grad[grad_name][slices]
                        )
                    else:
                        param[slices] -= local_grad

    def apply_accumulate_grad(self, beta=0.5):
        self.grad_normalization()

        self.model.to("cpu")

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                for local_grad, alpha in zip(self.local_grads, self.alphas):
                    local_param_grad = local_grad[name].cpu()
                    slices = tuple(
                        slice(0, min(sm_dim, lg_dim))
                        for sm_dim, lg_dim in zip(local_param_grad.shape, param.shape)
                    )
                    param[slices] -= (
                        local_param_grad * alpha / sum(self.alphas)
                    ) * beta

        self.local_grads.clear()
        self.alphas.clear()

    def train(
        self,
        args,
        data_shards,
        val_dataset,
        test_dataset=None,
        processor=None,
        collate_fn=None,
        compute_metrics=None,
    ):
        pass

    def grad_normalization(self):
        """Normalize the gradients via previous epoch's gradients"""
        pass

    def save_ckpt(self, dir):
        self.model.save_pretrained(os.path.join(dir))

    def load_ckpt(self, dir):
        self.model = self.model.from_pretrained(dir)
        # check the the existance of self.model.config.elastic_config
        assert hasattr(
            self.model.config, "elastic_config"
        ), "No elastic configuration found in the model config file. Please check the config file."


