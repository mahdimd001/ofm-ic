import os
import datetime
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import DataLoader
from ofm.modeling_ofm import OFM
from IC_NAS_Trainer import IC_NAS_Trainer
from arguments import arguments
from logger import init_logs, get_logger
from custom_transforms import transform, collate_fn
import evaluate
import functools
import timeit
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ofm.attention_pruning import OFM2


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Setup logging
    NOW = str(datetime.datetime.now()).replace(" ", "--")
    log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_dir = f'./logs/{NOW}_dataset[{args.dataset}]_trainable[{args.trainable}]_epochs[{args.epochs}]_lr[{args.lr}]_bs[{args.batch_size}]/'
    init_logs(log_file_name, args, log_dir)
    args.logger = get_logger()

    # Load dataset
    dataset = load_dataset(args.dataset, cache_dir=args.cache_dir)
    if args.dataset == "cifar100":
        dataset = dataset.rename_column("fine_label", "label")
    train_val = dataset["train"].train_test_split(test_size=0.2, seed=123)
    dataset["train"] = train_val["train"]
    dataset["validation"] = train_val["test"]
    labels = dataset["train"].features["label"].names

    # Initialize processor and dataset
    processor = AutoImageProcessor.from_pretrained(args.processor_name, cache_dir=args.cache_dir)
    #prepared_ds = dataset.with_transform(functools.partial(transform, processor=processor))

    def transform(example_batch, processor):
        inputs = processor([x.convert("RGB") for x in example_batch["img"]], return_tensors="pt")
        inputs["labels"] = example_batch["label"]
        return inputs

    prepared_ds = {
    "train": dataset["train"].with_transform(functools.partial(transform, processor=processor)),
    "validation": dataset["validation"].with_transform(functools.partial(transform, processor=processor)),
    }

    # Create data loaders
    train_dataloader = DataLoader(
        prepared_ds["train"], batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,persistent_workers=True, drop_last=True
    )
    test_dataloader = DataLoader(
        prepared_ds["validation"], batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,persistent_workers=True, drop_last=False
    )

    # Initialize model
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True,
        cache_dir=args.cache_dir,
    )

    # Define elastic config for NAS
    regular_config = {
        "atten_out_space": [768],
        "inter_hidden_space": [3072],
        "residual_hidden": [768],
    }
    elastic_config = {
        "atten_out_space": [768],
        "inter_hidden_space": [3072, 2048, 1024],
        "residual_hidden": [768],
    }
    config = {
        str(i): elastic_config if i in [1, 2, 5, 6, 9] else regular_config for i in range(12)
    }
    config["layer_elastic"] = {
        "elastic_layer_idx": [1, 2, 5, 6, 9],
        "remove_layer_prob": [0.5, 0.5, 0.5, 0.5, 0.5]
    }

    # Wrap model with OFM for NAS
    ofm = OFM(model.to("cpu"), elastic_config=config)
    testt = OFM2(model, elastic_config=config)
    args.supermodel = ofm
    args.pretrained = model

    args.logger.info(f'Original model size: {ofm.total_params} params')

    # Setup data loaders in args
    args.train_dataloader = train_dataloader
    args.test_dataloader = test_dataloader
    args.processor = processor
    args.device = device
    args.log_dir = log_dir

    # Initialize trainer
    trainer = IC_NAS_Trainer(args)

    # Evaluate pre-trained model
    start_test = timeit.default_timer()
    #accuracy, _, _ = trainer.eval(args.pretrained)
    accuracy = 44
    end_test = timeit.default_timer()
    args.logger.info(f'Pre-trained model size: {ofm.total_params} params \t Accuracy: {accuracy*100:.2f}% \t Time: {round(end_test - start_test, 4)} seconds')

    # Evaluate supernet
    start_test = timeit.default_timer()
    #accuracy, _, _ = trainer.eval(args.supermodel.model)
    accuracy = 44
    end_test = timeit.default_timer()
    args.logger.info(f'Supernet size: {ofm.total_params} params \t Pre-NAS Accuracy: {accuracy*100:.2f}% \t Time: {round(end_test - start_test, 4)} seconds')

    # Evaluate smallest submodel
    print("Evaluating smallest submodel")
    print(type(args.supermodel))
    submodel, params, arc = args.supermodel.smallest_model()
    start_test = timeit.default_timer()
    accuracy, _, _ = trainer.eval(submodel)
    end_test = timeit.default_timer()
    args.logger.info(f'Smallest model size: {params} params \t Pre-NAS Accuracy: {accuracy*100:.2f}% \t Time: {round(end_test - start_test, 4)} seconds')

    # Evaluate random submodel
    submodel, params, arc = args.supermodel.random_resource_aware_model()
    start_test = timeit.default_timer()
    accuracy, _, _ = trainer.eval(submodel)
    end_test = timeit.default_timer()
    args.logger.info(f'Medium model size: {params} params \t Pre-NAS Accuracy: {accuracy*100:.2f}% \t Time: {round(end_test - start_test, 4)} seconds')

    # Train with NAS
    args.logger.info('NAS Training starts')
    start = timeit.default_timer()
    trainer.train()
    end = timeit.default_timer()
    args.logger.info(f'NAS Training ends: {round(end - start, 4)} seconds')

if __name__ == "__main__":
    args = arguments()
    main(args)