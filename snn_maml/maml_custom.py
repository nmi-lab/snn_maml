from .maml import ModelAgnosticMetaLearning, batch_one_hot
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from snn_maml.utils import quantize_parameters
from collections import OrderedDict
from . import plasticity_rules
from .utils import tensors_to_device, compute_accuracy



class MAMLCustom(ModelAgnosticMetaLearning):
    def get_outer_loss(self, batch, hard_clamp = False):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        is_classification_task = (not test_targets.dtype.is_floating_point)
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.num_adaptation_steps,
                num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.
        }
        if is_classification_task:
            results.update({
                'accuracies_before': np.zeros((num_tasks,), dtype=np.float32),
                'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
            })

        mean_outer_loss = torch.tensor(0., device=self.device)
        # One task per batch_size
        for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
                in enumerate(zip(*batch['train'], *batch['test'])):
            
            params, adaptation_results = self.adapt(
                train_inputs, train_targets,
                is_classification_task=is_classification_task,
                num_adaptation_steps=self.num_adaptation_steps,
                step_size=self.step_size, first_order=self.first_order, hard_clamp=hard_clamp)

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            
            if is_classification_task:
                results['accuracies_before'][task_id] = adaptation_results['accuracy_before']

            with torch.set_grad_enabled(self.model.training):
                test_logits = self.model(test_inputs, params=params)
                if self.loss_function is F.mse_loss:
                    outer_loss = self.loss_function(test_logits, batch_one_hot(test_targets.to(test_targets.device), test_logits.shape[1]).cuda()) 
                else:
                    outer_loss = self.loss_function(test_logits[:,:,50:].mean(axis=2), test_targets)
                    
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

            if is_classification_task:
                results['accuracies_after'][task_id] = compute_accuracy(
                    test_logits[:,:,50:].mean(axis=2), test_targets)


        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        return mean_outer_loss, results
    
    #Inner loop
    def adapt(self, inputs, targets, is_classification_task=None,
              num_adaptation_steps=1, step_size=0.1, first_order=False, **kwargs):
        if is_classification_task is None:
            is_classification_task = (not targets.dtype.is_floating_point)
            
        params = OrderedDict(self.model.meta_named_parameters())
        if self.outer_loop_quantizer is not None:
            params = quantize_parameters(params, self.outer_loop_quantizer)

        results = {'inner_losses': np.zeros(
            (num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):
            
            logits = self.model(inputs, params=params)
            if self.loss_function == F.mse_loss:
                inner_loss = self.loss_function(logits, batch_one_hot(targets, logits.shape[1]).cuda()) 
            else:
                inner_loss = self.loss_function(logits[:,:,50:].mean(axis=2), targets)
            results['inner_losses'][step] = inner_loss.item()
            if (step == num_adaptation_steps-1) and is_classification_task: 
                results['accuracy_before'] = compute_accuracy(logits[:,:,50:].mean(axis=2), targets)

            #print("updating params...")
            self.model.zero_grad()
            if not self.soel:
                params = plasticity_rules.custom_sgd_layer(self.model,
                                                 inner_loss,
                                                 step_size=step_size,
                                                 params=params,
                                                 first_order=(not self.model.training) or first_order)
            elif self.soel:
                params = plasticity_rules.maml_soel_loihi(
                                   self.model,
                                   logits,
                                   targets,
                                   params=params,
                                   step_size=step_size,
                                   learning_steps=30,
                                   first_order=False,
                                   threshold = None)
             
        return params, results
    
    def train_iter(self, dataloader, max_batches=500, epoch=-1):
        if self.optimizer is None:
            raise RuntimeError('Trying to call `train_iter`, while the '
                'optimizer is `None`. In order to train `{0}`, you must '
                'specify a Pytorch optimizer as the argument of `{0}` '
                '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        num_batches = 0
        self.model.train()
        
        #print(self.model)
        
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break



                self.optimizer.zero_grad()

                batch = tensors_to_device(batch, device=self.device)
                outer_loss, results = self.get_outer_loss(batch)    
                yield results
                outer_loss.backward()
                if self.custom_outer_update_fn is not None:
                    self.custom_outer_update_fn(self.model)


                self.optimizer.step()
                if hasattr(self.step_size, '__len__'):
                    if len(self.step_size.shape)>0:
                        for name, value in self.step_size.items():
                            if value.data<0:
                                value.data.zero_()
                                print('Negative step values detected')
                                
                #if self.custom_outer_update_fn is not None:
                #    from .custom_funs import inplace_clamp_model_weights_asymm
                #    inplace_clamp_model_weights_asymm(self.model)
                
                # print(self.model.blocks[2].synapse.weight.max())

                if self.scheduler is not None:
                    self.scheduler.step()

                num_batches += 1


    def evaluate_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.model.eval()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                batch = tensors_to_device(batch, device=self.device)
                _, results = self.get_outer_loss(batch)
                yield results

                num_batches += 1

