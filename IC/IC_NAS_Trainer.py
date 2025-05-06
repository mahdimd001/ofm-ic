import torch
import torch.nn as nn
from tqdm import tqdm
from utility import get_trainable_parameters, get_optimizer_and_scheduler
import timeit
import copy
import evaluate
from ofm import OFM

class IC_NAS_Trainer:
    def __init__(self, args):
        for key, value in vars(args).items():
            setattr(self, key, value)
        self.best_smallest_submodel_accuracy = 0
        self.scheduler = None
        self.loss_func = nn.CrossEntropyLoss()

    def eval(self, model, map=None):
        accuracy_metric = evaluate.load("accuracy")
        model = model.eval()
        model = nn.DataParallel(model).to(self.device)
        predictions, labels = [], []
        for inputs in tqdm(self.test_dataloader, disable=self.no_verbose):
            torch.cuda.empty_cache()
            with torch.no_grad():
                outputs = model(pixel_values=inputs["pixel_values"].to(self.device))
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                labels.extend(inputs["labels"].cpu().numpy())
        model = model.module
        model = model.to('cpu')
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        return accuracy["accuracy"], None, map

    def training_step(self, model, inputs):
        local_grad = {k: v.cpu() for k, v in model.state_dict().items()}
        model = model.train()
        model = nn.DataParallel(model).to(self.device)
        self.optimizer.zero_grad()
        outputs = model(pixel_values=inputs["pixel_values"].to(self.device))
        loss = self.loss_func(outputs.logits, inputs["labels"].to(self.device))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        model = model.module
        model = model.to('cpu')
        with torch.no_grad():
            for k, v in model.state_dict().items():
                local_grad[k] = local_grad[k] - v.cpu()
        self.supermodel.apply_grad(local_grad, model.config.arch.get('remove_layer_idx', []))
        return {"train_loss": loss.item(), "params": model.config.num_parameters}

    def single_step(self, submodel, data, model_size, do_test):
        trainable_params = get_trainable_parameters(submodel, self.trainable)
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(
            trainable_params, self.lr, self.weight_decay, self.scheduler_type
        )
        start_train = timeit.default_timer()
        metrics = self.training_step(submodel, data)
        end_train = timeit.default_timer()
        if do_test:
            start_test = timeit.default_timer()
            accuracy, _, _ = self.eval(submodel)
            end_test = timeit.default_timer()
            if model_size == 'Smallest' and accuracy > self.best_smallest_submodel_accuracy:
                self.supermodel.save_ckpt(f'{self.log_dir}Best/')
                self.best_smallest_submodel_accuracy = accuracy
            metrics['test_accuracy'] = accuracy
            self.logger.info(
                f'\t{model_size} submodel train time: {round(end_train - start_train, 4)}, '
                f'test time: {round(end_test - start_test, 4)}, metrics: {metrics}'
            )
        else:
            self.logger.info(
                f'\t{model_size} submodel train time: {round(end_train - start_train, 4)}, metrics: {metrics}'
            )

    def train(self):
        for epoch in range(self.epochs):
            self.logger.info(f'EPOCH {epoch}: starts')
            if self.reorder == 'per_epoch':
                if self.reorder_method == 'magnitude':
                    self.supermodel.mlp_layer_reordering()
                elif self.reorder_method == 'wanda':
                    self.supermodel.mlp_layer_reordering(self.reorder_dataloader, 'wanda')
            start_epoch = timeit.default_timer()
            for idx, inputs in enumerate(tqdm(self.train_dataloader, disable=self.no_verbose)):
                if self.reorder == 'per_batch':
                    if self.reorder_method == 'magnitude':
                        self.supermodel.mlp_layer_reordering()
                    elif self.reorder_method == 'wanda':
                        self.supermodel.mlp_layer_reordering(self.reorder_dataloader, 'wanda')
                do_test = (idx == len(self.train_dataloader) - 1)
                save_interval = len(self.train_dataloader) // self.save_interval if self.save_interval else 1
                do_save = ((idx + 1) % save_interval == 0)
                do_test = do_test or do_save
                if 'l' in self.sandwich:
                    submodel, submodel.config.num_parameters, submodel.config.arch = (
                        copy.deepcopy(self.supermodel.model),
                        self.supermodel.total_params,
                        {'remove_layer_idx': []},
                    )
                    self.single_step(submodel, inputs, 'Largest', do_test)
                if 's' in self.sandwich:
                    submodel, submodel.config.num_parameters, submodel.config.arch = self.supermodel.smallest_model()
                    self.single_step(submodel, inputs, 'Smallest', do_test)
                if 'm' in self.sandwich:
                    submodel, submodel.config.num_parameters, submodel.config.arch = self.supermodel.random_resource_aware_model()
                    self.single_step(submodel, inputs, 'Medium', do_test)
                if do_save:
                    self.supermodel.save_ckpt(f'{self.log_dir}')
                    self.logger.info(f'\tInterval {(idx + 1) // save_interval}: Model checkpoint saved.')
            end_epoch = timeit.default_timer()
            self.supermodel.save_ckpt(f'{self.log_dir}')
            self.logger.info(f'EPOCH {epoch}: ends {round(end_epoch - start_epoch, 4)} seconds. Model checkpoint saved.')