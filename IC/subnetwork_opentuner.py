import numpy as np
from ofm import OFM
import torch
import time
import opentuner
from opentuner import ConfigurationManipulator, EnumParameter, MeasurementInterface, Result
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from arguments import arguments
from custom_transforms import transform, collate_fn
import evaluate
import functools

def arc_config_creator(
    atten_out_space,
    inter_hidden_space,
    residual_hidden_space,
    layer_elastic,
    n_layer=12,
    smallest=False,
    largest=False,
):
    arc_config = {}
    np.random.seed(int(time.time()))
    search_space = {}
    for layer in range(n_layer):
        conf_key = f"layer_{layer+1}_atten_out"
        search_space[conf_key] = atten_out_space
        conf_key = f"layer_{layer+1}_inter_hidden"
        search_space[conf_key] = inter_hidden_space
        conf_key = f"layer_{layer+1}_residual_hidden"
        search_space[conf_key] = residual_hidden_space
    return search_space

class NASOpenTuner(MeasurementInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = arguments()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load dataset
        dataset = load_dataset(self.args.dataset)
        if self.args.dataset == "cifar100":
            dataset = dataset.rename_column("fine_label", "label")
        train_val = dataset["train"].train_test_split(test_size=0.2, seed=123)
        dataset["train"] = train_val["train"]
        dataset["validation"] = train_val["test"]
        labels = dataset["train"].features["label"].names

        # Initialize processor and dataset
        self.processor = AutoImageProcessor.from_pretrained(self.args.processor_name)
        prepared_ds = dataset.with_transform(functools.partial(transform, processor=self.processor))
        self.test_dataloader = DataLoader(
            prepared_ds["validation"], batch_size=8, shuffle=False, collate_fn=collate_fn
        )

        # Initialize model
        self.model = AutoModelForImageClassification.from_pretrained(
            self.args.model_name,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)},
            ignore_mismatched_sizes=True,
        )
        self.ofm = OFM(self.model, elastic_config={
            "0": {
                "atten_out_space": [768, 512],
                "inter_hidden_space": [3072, 2048, 1024],
                "residual_hidden": [768]
            },
            "1": {
                "atten_out_space": [768, 512],
                "inter_hidden_space": [3072, 2048, 1024],
                "residual_hidden": [768]
            },
            # Add configs for other layers as needed
        })
        self.supermodel = self.ofm.model
        self.search_space = arc_config_creator(
            atten_out_space=self.supermodel.config.elastic_config['atten_out_space'],
            inter_hidden_space=self.supermodel.config.elastic_config['inter_hidden_space'],
            residual_hidden_space=self.supermodel.config.elastic_config['residual_hidden'],
            layer_elastic=[],
            n_layer=self.supermodel.vision_encoder.config.num_hidden_layers
        )

    def manipulator(self):
        manipulator = ConfigurationManipulator()
        for key in self.search_space:
            manipulator.add_parameter(EnumParameter(key, self.search_space[key]))
        return manipulator

    def run(self, desired_result, input, limit):
        cfg = desired_result.configuration.data
        arc_config = {}
        for layer in range(self.supermodel.vision_encoder.config.num_hidden_layers):
            attn_key = f"layer_{layer+1}_atten_out"
            inter_key = f"layer_{layer+1}_inter_hidden"
            residual_key = f"layer_{layer+1}_residual_hidden"
            arc_config[f"layer_{layer + 1}"] = {
                "atten_out": cfg[attn_key],
                "inter_hidden": cfg[inter_key],
                "residual_hidden": cfg[residual_key],
            }
        subnetwork, total_params = self.ofm.resource_aware_model(arc_config)

        # Evaluate subnetwork
        accuracy_metric = evaluate.load("accuracy")
        subnetwork = subnetwork.eval()
        subnetwork = nn.DataParallel(subnetwork).to(self.device)
        predictions, labels = [], []
        for inputs in self.test_dataloader:
            torch.cuda.empty_cache()
            with torch.no_grad():
                outputs = subnetwork(pixel_values=inputs["pixel_values"].to(self.device))
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                labels.extend(inputs["labels"].cpu().numpy())
        subnetwork = subnetwork.module
        subnetwork = subnetwork.to('cpu')
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]

        print(f'Subnetwork Config: {arc_config}')
        print(f'Params: {total_params}')
        print(f'Accuracy: {accuracy*100:.2f}%')
        return Result(time=(1 - accuracy))

    def save_final_config(self, configuration):
        print("Optimal configurations written to NAS_final_config.json:", configuration.data)
        self.manipulator().save_to_file(configuration.data, 'SAMNAS_final_config.json')

if __name__ == "__main__":
    argparser = opentuner.default_argparser()
    NASOpenTuner.main(argparser.parse_args())