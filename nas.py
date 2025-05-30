import os
import datetime
from transformers import SamModel, SamConfig, SamProcessor
from ofm import OFM
from torch.utils.data import DataLoader, Subset
from utility import *
from logger import init_logs, get_logger
from dataset import SA1BDataset, MitoDataset, COCOSegmentation, ADE20KDataset
#from coco_dataset import COCOSegmentation
import timeit
from COCO_NAS_Trainer import COCO_NAS_Trainer
from Mito_NAS_Trainer import Mito_NAS_Trainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


if __name__ == '__main__':

    DATA_ROOT = '/work/LAS/jannesar-lab/msamani/SuperSAM/datasets/' #'/dev/shm/abebe/datasets/' 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #torch.cuda.empty_cache()
    args = get_args()
    print(args)

    NOW = str(datetime.datetime.now()).replace(" ","--")
    log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_dir = './logs/{}_dataset[{}]_trainable[{}]_epochs[{}]_lr[{}]_local_bs[{}]/'. \
        format(NOW,args.dataset, args.trainable, args.epochs, args.lr, args.batch_size)
    
    #Use prompt_size 64 for SA1B and prompt_size 8 for COCO
    args.prompt_batch_size = 8 if args.dataset == 'coco' else 64
    

    init_logs(log_file_name, args, log_dir)
    args.logger = get_logger()

    #Initialize encoder
    # huge_model = SamModel.from_pretrained("facebook/sam-vit-huge")
    # huge_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    # Initialize the original SAM model and processor
    original_model = SamModel.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


    # layers_to_prune = [1,6,9]
    # ViT-B global_attn_indexes = [ 2, 5, 8, 11 ] #Taken from ViT config page on huggingface
    #ViT-L encoder_global_attn_indexes=[5, 11, 17, 23],
    #ViT-H encoder_global_attn_indexes=[7, 15, 23, 31],

    # assert ([1,5,7],[1,4,6,8]) == structured_pruning(original_model,layers_to_prune,global_attn_indexes), f'Prunning failed!'

    # OFM configuration and submodel initialization
    regular_config = {
        "atten_out_space": [768], #must be divisbly by num_heads==12
        "inter_hidden_space": [3072],
        "residual_hidden": [1020],
    }
    elastic_config = {
        "atten_out_space": [768], #Can't go over 768 [768,512]
        "inter_hidden_space": [768,1020,1536,2304], #Reduce for minimizing model size [256,768,1020,1536,2304]
        "residual_hidden": [1020],
    }

    config = {'0':regular_config, '1':elastic_config, '2':elastic_config,'3':elastic_config, '4':elastic_config,
                '5':elastic_config,'6':elastic_config,'7':elastic_config,'8':elastic_config,'9':elastic_config,
                '10':elastic_config,'11':elastic_config,
                "layer_elastic":{
                    # "elastic_layer_idx":  [],
                    # "remove_layer_prob":[]
                    #"elastic_layer_idx": [1,6,9],
                    #"remove_layer_prob":[.5,.5,.5]
                    "elastic_layer_idx":  [1,2,5,6,9],
                    "remove_layer_prob":[.5,.5,.5,.5,.5]
                }}



    ofm = OFM(original_model, elastic_config=config)

    args.supermodel = ofm
    args.pretrained = original_model

    args.logger.info(f'Original model size : {count_parameters(args.supermodel.model)} params')

    if args.dataset == 'mito':

        # Create an instance of the SAMDataset
        
        train_dataset = load_dataset(f'{DATA_ROOT}mitochondria/training.tif', f'{DATA_ROOT}mitochondria/training_groundtruth.tif')
        test_dataset = load_dataset(f'{DATA_ROOT}mitochondria/testing.tif', f'{DATA_ROOT}mitochondria/testing_groundtruth.tif')
        train_dataset = MitoDataset(dataset=train_dataset, processor=processor)
        test_dataset = MitoDataset(dataset=test_dataset, processor=processor)

        #Reordering dataset
        reordering_subset = Subset(train_dataset, indices=range(0,1 * args.batch_size,1))
        reorder_dataloader = DataLoader(reordering_subset, batch_size=2)
        args.reorder_dataloader = reorder_dataloader

        # Apply subset for shorter training
        if args.train_subset:
            subset_dataset = Subset(train_dataset, indices=range(args.train_subset))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

        
        # Apply subset for shorter testing
        if args.test_subset:
            subset_dataset = Subset(test_dataset, indices=range(args.test_subset))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        else:
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    
    elif args.dataset == 'sa1b':
        if 'e' in args.trainable and 'm' in args.trainable:
            train_dataset = SA1BDataset(f'{DATA_ROOT}SA1B', processor=processor,split='train',max_labels=args.prompt_batch_size)
        elif 'e' in args.trainable:
            train_dataset = SA1BDataset(f'{DATA_ROOT}SA1B', processor=processor,split='from_embedding')
        test_dataset = SA1BDataset(f'{DATA_ROOT}SA1B', processor=processor,split='val',max_labels=args.prompt_batch_size)

        #Use X batches of data for wanda reordering
        reordering_dataset = SA1BDataset(f'{DATA_ROOT}SA1B', processor=processor, split='train')
        subset_dataset = Subset(reordering_dataset, indices=range(0,2 * args.batch_size,1))
        reorder_dataloader = DataLoader(subset_dataset, batch_size=8, shuffle=True, drop_last=False, collate_fn=none_skipper_collate)
        args.reorder_dataloader = reorder_dataloader

        # Apply subset for shorter training
        if args.train_subset:
            subset_dataset = Subset(train_dataset, indices=range(0,args.train_subset,1))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=none_skipper_collate) #collate_fn = sa1b_collate_fn
        
        else:
            subset_dataset = Subset(train_dataset, indices=range(0,10000,1))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=none_skipper_collate)
        
        if args.test_subset:
            subset_dataset = Subset(test_dataset, indices=range(args.train_subset,args.train_subset + args.test_subset,1))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=none_skipper_collate) #collate_fn = sa1b_collate_fn
        else:
            subset_dataset = Subset(test_dataset, indices=range(10000,len(test_dataset),1))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=none_skipper_collate)

    elif args.dataset == 'ade20k':
        DATA_ROOT = 'ADE20K/ADEChallengeData2016/'
        train_dataset = ADE20KDataset(f'{DATA_ROOT}', processor=processor, split='training')
        test_dataset = ADE20KDataset(f'{DATA_ROOT}', processor=processor, split='validation')
 
        reordering_dataset = ADE20KDataset(f'{DATA_ROOT}', processor=processor, split='training')
        subset_dataset = Subset(reordering_dataset, indices=range(0,2 * args.batch_size,1))
        reorder_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=none_skipper_collate)

        # Apply subset for shorter training
        if args.train_subset:
            subset_dataset = Subset(train_dataset, indices=range(0,args.train_subset,1))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=none_skipper_collate) #collate_fn = custom_collate_fn
        
        else:
            #subset_dataset = Subset(train_dataset, indices=range(0,3000,1))
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=none_skipper_collate)
        
        if args.test_subset:
            subset_dataset = Subset(test_dataset, indices=range(0,args.test_subset,1))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=none_skipper_collate) #collate_fn = custom_collate_fn
        else:
            #subset_dataset = Subset(dataset, indices=range(3000,len(dataset),1))
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=none_skipper_collate) 



    elif args.dataset == 'coco':
        #dataset = COCOSegmentation('datasets/coco','val', processor=processor)
        args.base_size = 513
        args.crop_size = 513
        train_dataset = COCOSegmentation(args,f'{DATA_ROOT}coco','train', '2017', max_labels=args.prompt_batch_size, processor=processor)
        test_dataset = COCOSegmentation(args,f'{DATA_ROOT}coco','val', '2017', max_labels=args.prompt_batch_size, processor=processor)
        #test_dataset = COCOSegmentation('datasets/coco','val')

        reordering_dataset =  COCOSegmentation(args,f'{DATA_ROOT}coco','train', '2017', max_labels=args.prompt_batch_size, processor=processor)
        subset_dataset = Subset(reordering_dataset, indices=range(0,2 * args.batch_size,1))
        reorder_dataloader = DataLoader(subset_dataset, batch_size=8, shuffle=True, drop_last=False, collate_fn=none_skipper_collate) #none_skipper_collate
        args.reorder_dataloader = reorder_dataloader

        # Apply subset for shorter training
        if args.train_subset:
            subset_dataset = Subset(train_dataset, indices=range(0,args.train_subset,1))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=none_skipper_collate) #collate_fn = custom_collate_fn
        
        else:
            #subset_dataset = Subset(train_dataset, indices=range(0,3000,1))
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=none_skipper_collate)
        
        if args.test_subset:
            subset_dataset = Subset(test_dataset, indices=range(0,args.test_subset,1))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=none_skipper_collate) #collate_fn = custom_collate_fn
        else:
            #subset_dataset = Subset(dataset, indices=range(3000,len(dataset),1))
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=none_skipper_collate)

    #args.logger.info(f'Dataset : {len(dataset)}')
    args.logger.info(f'trainloader : {len(train_dataloader)} x batch_size : {args.batch_size}')
    args.logger.info(f'testloader : {len(test_dataloader)} x batch_size : {args.batch_size}')

    args.train_dataloader = train_dataloader
    args.test_dataloader = test_dataloader
    args.processor = processor
    args.device = device
    args.log_dir = log_dir

    #Reorder mlp layers to see if pretrained model is consistent
    if args.reorder == 'once':
        if args.reorder_method == 'magnitude':
            score_dist = args.supermodel.mlp_layer_reordering()
            plot_dist(score_dist,filename='magnitude-dist.png',importance='Magnitude')
        elif args.reorder_method == 'wanda':
            score_dist = args.supermodel.mlp_layer_reordering(reorder_dataloader,'wanda')
            plot_dist(score_dist,filename='wanda-dist.png',importance='Wanda')
        elif args.reorder_method == 'movement':
            score_dist = args.supermodel.mlp_layer_reordering(reorder_dataloader,'movement')
            plot_dist(score_dist,filename='movement-dist.png',importance='Movement')
        args.logger.info(f'Reordered {args.reorder} using {args.reorder_method if args.reorder_method else "No"} importance')

    #Initialize Trainer
    if args.dataset == 'mito':
        trainer = Mito_NAS_Trainer(args)
    else:
        trainer = COCO_NAS_Trainer(args)

    # Calculate IoUs and average IoU before training
    # start_test = timeit.default_timer()
    # miou, mious, map = trainer.eval(args.pretrained)    
    # end_test = timeit.default_timer()
    # #sorted_mious, indices = torch.sort(mious)
    # #args.logger.info(f'supermodel pre-NAS IoUs: {sorted_mious}')
    # args.logger.info(f'pre-trained model size : {count_parameters(original_model)} params \t mIoU : {miou}% \t time: {round(end_test - start_test,4)} seconds')
    # #save_preds(map,'Pretrained')


    start_test = timeit.default_timer()
    #miou, mious, map = eval(args.supermodel.model,test_dataloader,disable_verbose=args.no_verbose,processor=processor,prompt=args.test_prompt)
    miou, mious, map = trainer.eval(args.supermodel.model)
    end_test = timeit.default_timer()
    #sorted_mious, indices = torch.sort(mious)
    #args.logger.info(f'supermodel pre-NAS IoUs: {sorted_mious}')
    args.logger.info(f'supermodel size : {count_parameters(args.supermodel.model)} params \t pre-NAS mIoU : {miou}% \t time: {round(end_test - start_test,4)} seconds')
    #save_preds(map,'Largest')

    #save_preds(map,'Largest')
    submodel, submodel.config.num_parameters, submodel.config.arch = args.supermodel.smallest_model()
    start_test = timeit.default_timer()
    #miou, mious, map = eval(submodel,test_dataloader,disable_verbose=args.no_verbose,processor=processor,prompt=args.test_prompt)
    miou, mious, map = trainer.eval(submodel)
    end_test = timeit.default_timer()
    #sorted_mious, indices = torch.sort(mious)
    #args.logger.info(f'smallest pre-NAS IoUs: {sorted_mious}')
    args.logger.info(f'smallest model size : {count_parameters(submodel)} params \t  pre-NAS mIoU : {miou}% \t time: {round(end_test - start_test,4)} seconds')
    #save_preds(map,'Smallest')
    
    submodel, submodel.config.num_parameters, submodel.config.arch = args.supermodel.random_resource_aware_model()
    start_test = timeit.default_timer()
    #miou, mious, map = eval(submodel,test_dataloader,disable_verbose=args.no_verbose,processor=processor,prompt=args.test_prompt)
    miou, mious, map = trainer.eval(submodel)
    end_test = timeit.default_timer()
    #sorted_mious, indices = torch.sort(mious)
    #args.logger.info(f'medium pre-NAS IoUs: {sorted_mious}')
    args.logger.info(f'medium model size : {count_parameters(submodel)} params \t pre-NAS mIoU : {miou}% \t time: {round(end_test - start_test,4)} seconds')
    #save_preds(map,'Medium')


    args.logger.info(f'NAS Training starts')
    start = timeit.default_timer()

    trainer.train()
    
    #train_nas(args)
    
    end = timeit.default_timer()
    args.logger.info(f'NAS Training ends : {round(end-start,4)} seconds')