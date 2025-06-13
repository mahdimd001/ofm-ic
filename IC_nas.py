import datetime
import torch
from datasets import load_dataset,Image as DatasetImage
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import DataLoader, Subset
from ofm.modeling_ofm import OFM
from IC_NAS_Trainer import IC_NAS_Trainer
from IC_arguments import arguments
from logger import init_logs, get_logger
from IC_custom_transforms import transform, collate_fn
import functools
import timeit
import sys
from huggingface_hub import login
from PIL import Image
from io import BytesIO





def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            train_subset = dataset["train"].train_test_split(train_size=20000, seed=123, stratify_by_column="label")["train"]
            val_subset = dataset["validation"].train_test_split(train_size=5000, seed=123, stratify_by_column="label")["train"]
            dataset["train"] = train_subset
            dataset["validation"] = val_subset
        dataset['train'] = dataset['train'].rename_column("image", "img")
        dataset['validation'] = dataset['validation'].rename_column("image", "img")

        #dataset['train'] = dataset['train'].cast_column("img", DatasetImage(decode=False))
        #dataset['validation'] = dataset['validation'].cast_column("img", DatasetImage(decode=False))
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

    def transform(example_batch, processor):
        try:
            inputs = processor([x.convert("RGB") for x in example_batch["img"]], return_tensors="pt")
            inputs["labels"] = example_batch["label"]
            return inputs
        except Exception as e:
            print(f"Error processing batch: {e}")
            args.logger.warning(f"Error processing batch: {e}")
            # Handle the case where images cannot be processed
            # You can return a dummy input or skip this batch
            # For now, we will return an empty tensor
            inputs = processor([Image.new("RGB", (224, 224))], return_tensors="pt")
            inputs["labels"] = torch.tensor([-1])
            return inputs
        # images = []
        # labels = []

        # for img_info, label in zip(example_batch["img"], example_batch["label"]):
        #     try:
        #         img_bytes = img_info["bytes"]
        #         img = Image.open(BytesIO(img_bytes)).convert("RGB")
        #         images.append(img)
        #         labels.append(label)
        #     except Exception as e:
        #         print(f"Skipping corrupted image: {e}")
        #         args.logger.warning(f"Skipping corrupted image.")
                

        # if not images:
        #     # Return dummy image and label if all failed
        #     dummy_input = processor([Image.new("RGB", (224, 224))], return_tensors="pt")
        #     dummy_input["labels"] = torch.tensor([-1])
        #     args.logger.warning(f"all images are corrupted, returning dummy input.")
        #     return dummy_input

        # inputs = processor(images, return_tensors="pt")
        # inputs["labels"] = torch.tensor(labels)
        # return inputs

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
        '/work/LAS/jannesar-lab/msamani/bestmodel/best_model_iter2.pt',
        #args.model_name,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        use_safetensors=True
    )


    # Reordering dataset
    # Create a subset and apply the same transform as the main dataset
    reordering_subset = dataset["train"].select(range(0, 20 * args.batch_size, 1)).with_transform(
        functools.partial(transform, processor=processor)
    )
    reorder_dataloader = DataLoader(
        reordering_subset, 
        batch_size=args.batch_size,  # Match main batch size for consistency
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
            "inter_hidden_space": [384, 576, 768],
            "residual_hidden": [192],
        }
        config = {
            str(i): elastic_config if i in [1,2, 3,4,5,6,7,8,9,10] else regular_config for i in range(12)
        }
        config["layer_elastic"] = {
            "elastic_layer_idx": [5,10],
            "remove_layer_prob": [.5, 0.5]
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
            "elastic_layer_idx": [5,7,10],
            "remove_layer_prob": [.5, 0.5, 0.5]
        }
    print("loading ofm model...")
    # Wrap model with OFM for NAS
    ofm = OFM(model.to("cpu"), elastic_config=config)
    args.supermodel = ofm
    args.pretrained = model
    print("load model to cpu finished.")
    args.logger.info(f'Original model size: {ofm.total_params} params')

    # Setup data loaders in args
    args.train_dataloader = train_dataloader
    args.test_dataloader = test_dataloader
    args.processor = processor
    args.device = device
    args.log_dir = log_dir

    # Initialize trainer
    trainer = IC_NAS_Trainer(args)
    print("stat evaluation...")
    # Evaluate pre-trained model
    start_test = timeit.default_timer()
    accuracy, _, _ = trainer.eval(args.pretrained)
    end_test = timeit.default_timer()
    args.logger.info(f'Pre-trained model size: {ofm.total_params} params \t Accuracy: {accuracy*100:.2f}% \t Time: {round(end_test - start_test, 4)} seconds')

    
    for i in range(args.finetune_epoches):
        args.logger.info(f'Fine-tuning epoch {i+1}/{args.finetune_epoches}')
        start_finetune = timeit.default_timer()
        trainer.fine_tune(args.finetune_epoches)
        end_finetune = timeit.default_timer()
        args.logger.info(f'Fine-tuning epoch {i+1} completed in {round(end_finetune - start_finetune, 4)} seconds, Accuracy: {accuracy*100:.2f}%')
    
    
    
    #Evaluate supernet
    # start_test = timeit.default_timer()
    # accuracy, _, _ = trainer.eval(args.supermodel.model)
    # end_test = timeit.default_timer()
    # args.logger.info(f'Supernet size: {ofm.total_params} params \t Pre-NAS Accuracy: {accuracy*100:.2f}% \t Time: {round(end_test - start_test, 4)} seconds')

    # Evaluate smallest submodel
    # submodel, submodel.config.num_parameters, submodel.config.arch = args.supermodel.smallest_model()
    # start_test = timeit.default_timer()
    # accuracy, _, _ = trainer.eval(submodel)
    # end_test = timeit.default_timer()
    # args.logger.info(f'Smallest model size: {submodel.config.num_parameters} params \t Pre-NAS Accuracy: {accuracy*100:.2f}% \t Time: {round(end_test - start_test, 4)} seconds')

    # Evaluate random submodel
    # submodel, submodel.config.num_parameters, submodel.config.arch = args.supermodel.random_resource_aware_model()
    # start_test = timeit.default_timer()
    # accuracy, _, _ = trainer.eval(submodel)
    # end_test = timeit.default_timer()
    # args.logger.info(f'Medium model size: {submodel.config.num_parameters} params \t Pre-NAS Accuracy: {accuracy*100:.2f}% \t Time: {round(end_test - start_test, 4)} seconds')

    # Train with NAS
    args.logger.info('NAS Training starts')
    start = timeit.default_timer()
    trainer.train()
    end = timeit.default_timer()
    args.logger.info(f'NAS Training ends: {round(end - start, 4)} seconds')

if __name__ == "__main__":
    args = arguments()
    main(args)
