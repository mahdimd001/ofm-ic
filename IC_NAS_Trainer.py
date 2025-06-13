import torch
import torch.nn as nn
from tqdm import tqdm
from utility import get_trainable_parameters, get_optimizer_and_scheduler
import timeit
import copy
import evaluate
from ofm import OFM
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np
from torchvision.transforms import Resize
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis
from thop import profile
from ofm.weight_reorder import compute_global_ffn_allocation

class IC_NAS_Trainer:
    def __init__(self, args):
        for key, value in vars(args).items():
            setattr(self, key, value)
        self.best_smallest_submodel_accuracy = 0
        self.scheduler = None
        self.loss_func = nn.CrossEntropyLoss()

        #tensorboard writer
        self.tensorboard_dir = f'{self.log_dir}/tensorboard'
        self.writer = SummaryWriter(self.tensorboard_dir)
        self.logger.info(f"TensorBoard logs saved to: {self.tensorboard_dir}")
        
    def _log_parameter_counts(self):
        """Log the number of parameters in each part of the model."""
        model = self.supermodel.model
        embedding_params = sum(p.numel() for p in model.vit.embeddings.parameters())
        encoder_params = sum(p.numel() for p in model.vit.encoder.parameters())
        classifier_params = sum(p.numel() for p in model.classifier.parameters())
        total_params = embedding_params + encoder_params + classifier_params
        self.logger.info(
            f"Model parameter counts:\n"
            f"  vit.embedding: {embedding_params:,} parameters\n"
            f"  vit.encoder: {encoder_params:,} parameters\n"
            f"  classifier: {classifier_params:,} parameters\n"
            f"  Total: {total_params:,} parameters"
        )
    
    def count_intermediate_outputs(self,model):
        """
        Counts the number of weights (params) in the intermediate FFN layer of each ViT encoder block.
        Assumes HuggingFace ViT model structure.
        
        Returns:
            results (list): List of out_features for each intermediate layer.
            total_weights (int): Total number of outputs in all intermediate layers.
        """
        model.eval()
        model = model.to('cpu')
        
        results = []
        total_weights = 0

        for i, layer in enumerate(model.vit.encoder.layer):
            intermediate = layer.intermediate.dense
            out_features = intermediate.out_features

            # Count weights: weights + bias
            total_weights += out_features

            results.append(out_features)

        return results, total_weights
    

    def _compute_flops3(self, model):
        """Compute forward pass FLOPs for the model in GFLOPs, dataset-agnostic."""
        model.eval()
        model = model.to('cpu')

        # Determine input resolution
        if hasattr(self, 'processor') and hasattr(self.processor, 'image_size'):
            if isinstance(self.processor.image_size, dict):
                height = self.processor.image_size.get('height', 224)
                width = self.processor.image_size.get('width', 224)
            else:
                height = width = self.processor.image_size
        elif hasattr(self, 'image_size'):
            height = width = self.image_size
        else:
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

    def _compute_flops2(self, model):
        """Compute forward pass FLOPs for the model in GFLOPs, dataset-agnostic."""
        model.eval()
        model = model.to('cpu')  # fvcore works on CPU

        # Determine input resolution from processor or args
        if hasattr(self, 'processor') and hasattr(self.processor, 'image_size'):
            if isinstance(self.processor.image_size, dict):
                height = self.processor.image_size.get('height', 224)
                width = self.processor.image_size.get('width', 224)
            else:
                height = width = self.processor.image_size
        elif hasattr(self, 'image_size'):
            height = width = self.image_size
        else:
            height = width = 224  # Default for ViT-Base
        input_shape = (1, 3, height, width)  # Batch=1 for FLOPs computation

        # Compute total parameters
        params = sum(p.numel() for p in model.parameters())

        # Compute forward pass FLOPs using fvcore
        input_tensor = torch.randn(input_shape)
        flop_analyzer = FlopCountAnalysis(model, input_tensor)
        flop_analyzer.uncalled_modules_warnings(False)  # Suppress warnings for uncalled modules
        flops = flop_analyzer.total()
        gflops = flops / 1e9  # Convert to GFLOPs


        return params, gflops
    def _compute_flops(self, model):
        """Compute FLOPs for the model in GFLOPs, dataset-agnostic."""
        model.eval()
        model = model.to('cpu')  # ptflops works on CPU

        # Determine input resolution from processor or args
        if hasattr(self, 'processor') and hasattr(self.processor, 'image_size'):
            if isinstance(self.processor.image_size, dict):
                height = self.processor.image_size.get('height', 224)
                width = self.processor.image_size.get('width', 224)
            else:
                height = width = self.processor.image_size
        elif hasattr(self, 'image_size'):
            height = width = self.image_size
        else:
            height = width = 224  # Default for ViT-Base
        input_res = (3, height, width)
        #self.logger.info(f"Computing FLOPs with input resolution: {input_res}")

        macs, params = get_model_complexity_info(
            model,
            input_res,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=True
        )
        flops = macs  # MACs to FLOPs (multiply by 2 for add and multiply)
        gflops = flops / 1e9  # Convert to GFLOPs
        return params,gflops

    def visualize_predictions(self, model, model_name, step):
        model.eval()
        model = nn.DataParallel(model).to(self.device)
        images_to_log = []
        max_images = 10
        resize = Resize((224, 224))  # Resize to 224x224

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_dataloader):
                if batch_idx > 0:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(pixel_values=batch["pixel_values"])
                preds = torch.argmax(outputs.logits, dim=1)
                images = batch["pixel_values"][:max_images]
                gt_indices = batch["labels"][:max_images].cpu().numpy()
                pred_indices = preds[:max_images].cpu().numpy()

                mean = torch.tensor(self.processor.image_mean).view(1, 3, 1, 1).to(self.device)
                std = torch.tensor(self.processor.image_std).view(1, 3, 1, 1).to(self.device)
                images = images * std + mean
                images = images.clamp(0, 1)

                for i in range(min(max_images, images.size(0))):
                    img = images[i]
                    gt_text = self.labels[gt_indices[i]] if hasattr(self, 'labels') and self.labels else f"Class {gt_indices[i]}"
                    pred_text = self.labels[pred_indices[i]] if hasattr(self, 'labels') and self.labels else f"Class {pred_indices[i]}"

                    # Ground truth text image
                    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
                    ax.text(0.5, 0.5, gt_text, ha='center', va='center', fontsize=12)
                    ax.axis('off')
                    plt.tight_layout()
                    gt_img = self.fig_to_tensor(fig).to(self.device)
                    gt_img = resize(gt_img)  # Resize to [3, 224, 224]
                    plt.close(fig)

                    # Predicted text image
                    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
                    ax.text(0.5, 0.5, pred_text, ha='center', va='center', fontsize=12)
                    ax.axis('off')
                    plt.tight_layout()
                    pred_img = self.fig_to_tensor(fig).to(self.device)
                    pred_img = resize(pred_img)  # Resize to [3, 224, 224]
                    plt.close(fig)

                    images_to_log.extend([img, gt_img, pred_img])

        if images_to_log:
            grid = make_grid(images_to_log, nrow=3, padding=5, normalize=True)
            self.writer.add_image(f"{model_name}/Predictions", grid, step)

        model = model.module
        model = model.to('cpu')

    def fig_to_tensor(self, fig):
        """Convert matplotlib figure to tensor."""
        fig.canvas.draw()
        # Get RGBA buffer and convert to RGB
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # [H, W, 4]
        img = buf[:, :, :3]  # Discard alpha channel [H, W, 3]
        img = img.transpose(2, 0, 1)  # [C, H, W]
        img = torch.from_numpy(img).float() / 255.0  # [C, H, W], 0-1 range
        return img

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

    def training_step(self, model, inputs, teacher_logits=None, model_size=None):
        local_grad = {k: v.cpu() for k, v in model.state_dict().items()}
        model = model.train()
        model = nn.DataParallel(model).to(self.device)
        self.optimizer.zero_grad()
        outputs = model(pixel_values=inputs["pixel_values"].to(self.device))
        logits = outputs.logits.detach().cpu()
        loss = self.loss_func(outputs.logits, inputs["labels"].to(self.device))

        # if self.enable_KD and teacher_logits is not None:
        #     kd_loss = nn.MSELoss()(outputs.logits, teacher_logits.to(self.device))
        #     loss = loss + kd_loss

        #  Distillation Loss (Only for Small/Medium)
        if teacher_logits is not None:
            T = 2.0  # temperature
            student_probs = nn.functional.log_softmax(outputs.logits / T, dim=-1)
            teacher_probs = nn.functional.softmax(teacher_logits / T, dim=-1)
            teacher_probs = teacher_probs.to(self.device)
            student_probs = student_probs.to(self.device)
            distill_loss = nn.KLDivLoss(reduction='batchmean')(student_probs, teacher_probs) * (T ** 2)
            
            alpha = 0.5  # weighting between real and distill loss
            loss = alpha * loss + (1 - alpha) * distill_loss



        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        model = model.module
        model = model.to('cpu')
        with torch.no_grad():
            for k, v in model.state_dict().items():
                local_grad[k] = local_grad[k] - v.cpu()
        self.supermodel.apply_grad(local_grad,model.config.arch['remove_layer_idx'])
        return {"train_loss": loss.item(), "params": model.config.num_parameters},logits



    def fine_tuning_step(self, model, inputs):
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
        self.supermodel.apply_grad(local_grad)
        return {"train_loss": loss.item()}

    def single_step(self, submodel, data, model_size, do_test,teacher_logits=None):
        trainable_params = get_trainable_parameters(submodel, self.trainable)
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(
            trainable_params, self.lr, self.weight_decay, self.scheduler
        )
        start_train = timeit.default_timer()
        metrics,logits = self.training_step(submodel, data, teacher_logits,model_size)
        end_train = timeit.default_timer()
        # Compute FLOPs
        param2,flops = self._compute_flops3(submodel)
        metrics['params2'] = param2
        metrics['flops'] = flops

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
                f'test time: {round(end_test - start_test, 4)}, '
                f'metrics: {{train_loss: {metrics["train_loss"]:.4f}, '
                f'params: {metrics["params"]:,}, '
                f'params2: {metrics["params2"]:,}, '
                f'flops: {metrics["flops"]:.2f} GFLOPs, '
                f'test_accuracy: {metrics["test_accuracy"]:.4f}}}'
            )
        else:
            self.logger.info(
                f'\t{model_size} submodel train time: {round(end_train - start_train, 4)}, '
                f'metrics: {{train_loss: {metrics["train_loss"]:.4f}, '
                f'params: {metrics["params"]:,}, '
                f'params2: {metrics["params2"]:,}, '
                f'flops: {metrics["flops"]:.2f} GFLOPs}}'
            )
        return logits

    def train(self):
        global_step = 0
        for epoch in range(self.epochs):
            self.logger.info(f'EPOCH {epoch}: starts')
            if self.reorder == 'per_epoch':
                if self.reorder_method == 'magnitude':
                    self.supermodel.mlp_layer_reordering()
                elif self.reorder_method == 'wanda':
                    self.supermodel.mlp_layer_reordering(self.reorder_dataloader, 'wanda')
            start_epoch = timeit.default_timer()
            for idx, inputs in enumerate(tqdm(self.train_dataloader, disable=self.no_verbose)):
                global_step += 1
                if self.reorder == 'per_batch':
                    if self.reorder_method == 'magnitude':
                        self.supermodel.mlp_layer_reordering()
                    elif self.reorder_method == 'wanda':
                        self.supermodel.mlp_layer_reordering(self.reorder_dataloader, 'wanda')
                do_test = (idx == len(self.train_dataloader) - 1)
                save_interval = len(self.train_dataloader) // self.save_interval if self.save_interval else 1
                do_save = ((idx + 1) % save_interval == 0)

                do_test = do_test or do_save
                submodels = []
                teacher_logits = None
                if 'l' in self.sandwich:
                    submodel, submodel.config.num_parameters, submodel.config.arch = (
                        copy.deepcopy(self.supermodel.model),
                        self.supermodel.total_params,
                        {'remove_layer_idx': []},
                    )
                    teacher_logits = self.single_step(submodel, inputs, 'Largest', do_test)
                    if not self.enable_KD:
                        teacher_logits = None

                    submodels.append(('Largest', submodel))
                if 's' in self.sandwich:
                    submodel, submodel.config.num_parameters, submodel.config.arch = self.supermodel.smallest_model()
                    self.single_step(submodel, inputs, 'Smallest', do_test, teacher_logits)
                    submodels.append(('Smallest', submodel))
                if 'm' in self.sandwich:
                    submodel, submodel.config.num_parameters, submodel.config.arch = self.supermodel.random_resource_aware_model()
                    param1 = submodel.config.arch
                    self.single_step(submodel, inputs, 'Medium', do_test, teacher_logits)
                    submodels.append(('Medium', submodel))


                    if do_test:

                        a1,a2 = self.count_intermediate_outputs(submodel)
                        a3 = compute_global_ffn_allocation(self.supermodel.model, target_param=a2,compute_score='magnitude',removed_layers=submodel.config.arch['remove_layer_idx'])
                        self.logger.info(f"Global FFN allocation: {str(a3.items())}")

                        #a4 = compute_global_ffn_allocation(self.supermodel.model, a2,'wanda',self.reorder_dataloader)
                        # create a model based on the a4

                        submodel, submodel.config.num_parameters, submodel.config.arch = self.supermodel.smart_resource_aware_model(a3,submodel.config.arch['remove_layer_idx'])
                        # param2 = submodel.config.arch
                        # self.single_step(submodel, inputs, 'Smart', do_test)
                        accuracy , _, _ =self.eval(submodel)
                        self.logger.info(f'\tSmart submodel accuracy: {accuracy:.4f}')


                    # def compare_inter_hidden(param1, param2):
                    #     print(f"{'Layer':^8} | {'Param1':^10} | {'Param2':^10} | {'Difference':^10}")
                    #     print("-" * 44)
                    #     for i in range(12):  # layers 0 to 11
                    #         layer = str(i)
                    #         val1 = param1[layer]['inter_hidden']
                    #         val2 = param2[layer]['inter_hidden']
                    #         diff = val2 - val1
                    #         diff_str = f"{diff:+}"
                    #         print(f"{layer:^8} | {val1:^10} | {val2:^10} | {diff_str:^10}")

                    # if do_test:
                    #     compare_inter_hidden(param1, param2)


                if do_save:
                    self.supermodel.save_ckpt(f'{self.log_dir}')
                    self.logger.info(f'\tInterval {(idx + 1) // save_interval}: Model checkpoint saved.')
                    
                    if self.tensorboard_visual:
                        for model_name, submodel in submodels:
                            self.visualize_predictions(submodel, model_name, global_step)

            end_epoch = timeit.default_timer()
            self.supermodel.save_ckpt(f'{self.log_dir}')
            self.logger.info(f'EPOCH {epoch}: ends {round(end_epoch - start_epoch, 4)} seconds. Model checkpoint saved.')
        # Close TensorBoard writer
        self.writer.close()


    def fine_tune(self, epochs):
        global_step = 0
        best_accuracy = 0.0  # Track the best accuracy to save the best checkpoint

        # Define evaluation interval (e.g., evaluate every 10% of the dataloader)
        eval_interval = max(1, len(self.train_dataloader) // 10)  # Evaluate 10 times per epoch

        # Initialize optimizer and scheduler for the supermodel
        trainable_params = get_trainable_parameters(self.supermodel.model, self.trainable)
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(
            trainable_params, self.lr, self.weight_decay, self.scheduler
        )

        for epoch in range(epochs):
            self.logger.info(f'EPOCH {epoch}: starts')
            start_epoch = timeit.default_timer()

            # Set model to training mode
            self.supermodel.model.train()
            self.supermodel.model = self.supermodel.model.to(self.device)
            

            # Training loop
            total_loss = 0.0
            for idx, inputs in enumerate(tqdm(self.train_dataloader, disable=self.no_verbose)):
                global_step += 1
                inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to device

                # Perform a training step
                metrics = self.fine_tuning_step(self.supermodel.model, inputs)
                total_loss += metrics["train_loss"]

                # Evaluate accuracy at specified intervals within the epoch
                if (idx + 1) % eval_interval == 0 or idx == len(self.train_dataloader) - 1:
                    self.supermodel.model.eval()
                    accuracy, _, _ = self.eval(self.supermodel.model)
                    self.supermodel.model.train()  # Switch back to training mode
                    self.logger.info(f'\tStep {global_step} (Batch {idx + 1}/{len(self.train_dataloader)}): Accuracy: {accuracy*100:.2f}')

                # Save checkpoint at specified intervals or at the end of the epoch
                save_interval = len(self.train_dataloader) // self.save_interval if self.save_interval else 1
                if (idx + 1) % save_interval == 0 or idx == len(self.train_dataloader) - 1:
                    self.supermodel.save_ckpt(f'{self.log_dir}')
                    self.logger.info(f'\tStep {global_step}: Model checkpoint saved.')

            # Compute average loss for the epoch
            avg_loss = total_loss / len(self.train_dataloader)
            self.logger.info(f'\tEpoch {epoch} Average Loss: {avg_loss:.2f}')

            # Final evaluation at the end of the epoch
            self.supermodel.model.eval()
            accuracy, _, _ = self.eval(self.supermodel.model)
            self.logger.info(f'\tEpoch {epoch} Final Accuracy: {accuracy*100:.2f}')

            # Save the best model based on accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.supermodel.save_ckpt(f'{self.log_dir}best/')
                self.logger.info(f'\tNew best accuracy: {best_accuracy:.2f}, saved checkpoint to {self.log_dir}best/')

            end_epoch = timeit.default_timer()
            self.logger.info(f'EPOCH {epoch}: ends {round(end_epoch - start_epoch, 4)} seconds.')

        # Close TensorBoard writer
        self.writer.close()