from opentuner.search.manipulator import ConfigurationManipulator, IntegerParameter, EnumParameter
from opentuner.measurement import MeasurementInterface
from opentuner.resultsdb.models import Result
from transformers import ViTForImageClassification
from utility import structured_pruning_vit
import torch
import numpy as np
import copy

class NASTuner(MeasurementInterface):
    def __init__(self, args, supernet, trainer):
        super(NASTuner, self).__init__(args=args)
        self.supernet = supernet
        self.trainer = trainer
        self.max_layers_to_prune = 6  # Max 6 layers pruned (min 6 layers)
        self.elastic_layers = [1, 2, 4, 5, 7, 8, 9, 10]  # From config
        self.mlp_sizes = [2304, 1536, 1020, 768]  # From inter_hidden_space
        self.parallel_compile = True

    def manipulator(self):
        """Define the search space for NAS."""
        manipulator = ConfigurationManipulator()
        # Number of layers to prune (0 to 6)
        manipulator.add_parameter(IntegerParameter("num_layers_to_prune", 0, self.max_layers_to_prune))
        # MLP hidden size
        manipulator.add_parameter(EnumParameter("mlp_hidden_size", self.mlp_sizes))
        return manipulator

    def run(self, desired_result, input, limit):
        """Evaluate a configuration (architecture)."""
        cfg = desired_result.configuration.data
        num_layers_to_prune = cfg["num_layers_to_prune"]
        mlp_hidden_size = cfg["mlp_hidden_size"]

        # Create submodel
        submodel = copy.deepcopy(self.supernet.model)  # Deep copy to avoid modifying supernet
        remove_layer_idx = []
        if num_layers_to_prune > 0:
            # Randomly select layers to prune from elastic_layers
            remove_layer_idx = np.random.choice(self.elastic_layers, num_layers_to_prune, replace=False).tolist()
            submodel, _ = structured_pruning_vit(submodel, remove_layer_idx)

        # Scale MLP hidden size
        if mlp_hidden_size != 3072:  # Default ViT-Base MLP size
            for layer in submodel.vit.encoder.layer:
                # Trim or adjust MLP weights
                layer.intermediate.dense.weight = torch.nn.Parameter(
                    layer.intermediate.dense.weight[:, :mlp_hidden_size]
                )
                layer.intermediate.dense.out_features = mlp_hidden_size
                layer.output.dense.weight = torch.nn.Parameter(
                    layer.output.dense.weight[:mlp_hidden_size, :]
                )
                layer.output.dense.in_features = mlp_hidden_size

        # Update config for OFM
        submodel.config.arch = {"remove_layer_idx": remove_layer_idx}
        submodel.config.num_parameters = sum(p.numel() for p in submodel.parameters())

        # Evaluate submodel
        accuracy, _, _ = self.trainer.eval(submodel)
        params, gflops = self.trainer._compute_flops3(submodel)  # Use thop for consistency

        # Multi-objective: maximize accuracy, minimize GFLOPs
        normalized_accuracy = accuracy
        normalized_gflops = gflops / 20.0  # Normalize based on max 20 GFLOPs
        objective = 0.7 * normalized_accuracy - 0.3 * normalized_gflops  # Weight accuracy higher

        return Result(time=0, accuracy=accuracy, gflops=gflops, objective=objective)

    def save_results(self, results_db, configuration, result):
        """Save results to database."""
        results_db.results.append(result)