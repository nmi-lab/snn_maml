import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from snn_maml.utils import quantize_parameters
from collections import OrderedDict
from . import plasticity_rules
from .utils import tensors_to_device, compute_accuracy

__all__ = ['ModelAgnosticMetaLearning', 'MAML', 'FOMAML']

from tensorboardX import SummaryWriter

import pdb

# default `log_dir` is "runs" - we'll be more specific here

def batch_one_hot(targets, num_classes=10):
    one_hot = torch.zeros((targets.shape[0],num_classes))
    #print("targets shape", targets.shape)
    for i in range(targets.shape[0]):
        one_hot[i][targets[i]] = 1
        
    return one_hot

def undo_onehot(targets):
    not_hot = torch.zeros((targets.shape[0]))
    
    for i in range(targets.shape[0]):
        not_hot[i] = torch.nonzero(targets[0])[0][0].item()
        
    return not_hot.to(targets.device)


class ModelAgnosticMetaLearning(object):
    """Meta-learner class for Model-Agnostic Meta-Learning [1].

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    optimizer : `torch.optim.Optimizer` instance, optional
        The optimizer for the outer-loop optimization procedure. This argument
        is optional for evaluation.

    step_size : float (default: 0.1)
        The step size of the gradient descent update for fast adaptation
        (inner-loop update).

    first_order : bool (default: False)
        If `True`, then the first-order approximation of MAML is used.

    learn_step_size : bool (default: False)
        If `True`, then the step size is a learnable (meta-trained) additional
        argument [2].

    per_param_step_size : bool (default: False)
        If `True`, then the step size parameter is different for each parameter
        of the model. Has no impact unless `learn_step_size=True`.

    num_adaptation_steps : int (default: 1)
        The number of gradient descent updates on the loss function (over the
        training dataset) to be used for the fast adaptation on a new task.

    scheduler : object in `torch.optim.lr_scheduler`, optional
        Scheduler for the outer-loop optimization [3].

    loss_function : callable (default: `torch.nn.functional.cross_entropy`)
        The loss function for both the inner and outer-loop optimization.
        Usually `torch.nn.functional.cross_entropy` for a classification
        problem, of `torch.nn.functional.mse_loss` for a regression problem.

    device : `torch.device` instance, optional
        The device on which the model is defined.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Li Z., Zhou F., Chen F., Li H. (2017). Meta-SGD: Learning to Learn
           Quickly for Few-Shot Learning. (https://arxiv.org/abs/1707.09835)

    .. [3] Antoniou A., Edwards H., Storkey A. (2018). How to train your MAML.
           International Conference on Learning Representations (ICLR).
           (https://arxiv.org/abs/1810.09502)
    """
    def __init__(self, model, optimizer=None, step_size=0.1, first_order=False,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=F.cross_entropy,
                 custom_outer_update_fn = None, custom_inner_update_fn = None,
                 device=None, 
                 boil=False,
                 outer_loop_quantizer = None,
                 inner_loop_quantizer = None):
        self.model = model.to(device=device)
        self.outer_loop_quantizer = outer_loop_quantizer 
        self.inner_loop_quantizer = inner_loop_quantizer 
        self.optimizer = optimizer
        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.custom_inner_update_fn = custom_inner_update_fn
        self.custom_outer_update_fn = custom_outer_update_fn

        if per_param_step_size or boil:
            self.step_size = OrderedDict((name, torch.tensor(step_size, dtype=param.dtype, device=self.device, requires_grad=learn_step_size)) for (name, param) in model.meta_named_parameters())
            if boil:
                assert learn_step_size is False, 'boil is not compatible with learning step sizes'
                last_layer_names = [k for k in self.step_size.keys()][-2:]#assumed bias and weight in last layer
                for k in last_layer_names:
                    self.step_size[k] = torch.tensor(0., dtype=self.step_size[k].dtype, device=self.device, requires_grad=False)
                print('step_size', self.step_size)
        else:
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                device=self.device, requires_grad=learn_step_size)

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values() if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                #self.scheduler.base_lrs([group['initial_lr'] for group in self.optimizer.param_groups])

    def get_outer_loss(self, batch):
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
            
           # print("INPUT SHAPE", train_inputs.shape)
            
            params, adaptation_results = self.adapt(
                train_inputs, train_targets,
                is_classification_task=is_classification_task,
                num_adaptation_steps=self.num_adaptation_steps,
                step_size=self.step_size, first_order=self.first_order)

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            
            if is_classification_task:
                results['accuracies_before'][task_id] = adaptation_results['accuracy_before']

            with torch.set_grad_enabled(self.model.training):
                test_logits = self.model(test_inputs, params=params)
                outer_loss = self.loss_function(test_logits, test_targets)
                    
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

            if is_classification_task:
                results['accuracies_after'][task_id] = compute_accuracy(
                    test_logits, test_targets)


        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        return mean_outer_loss, results
    
    #Inner loop
    def adapt(self, inputs, targets, is_classification_task=None,
              num_adaptation_steps=1, step_size=0.1, first_order=False):
        if is_classification_task is None:
            is_classification_task = (not targets.dtype.is_floating_point)
            
        params = OrderedDict(self.model.meta_named_parameters())
        if self.outer_loop_quantizer is not None:
            params = quantize_parameters(params, self.outer_loop_quantizer)

        results = {'inner_losses': np.zeros(
            (num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):
            
            logits = self.model(inputs, params=params)

            inner_loss = self.loss_function(logits, targets)
            results['inner_losses'][step] = inner_loss.item()
            #pdb.set_trace()
            if (step == num_adaptation_steps-1) and is_classification_task: 
                results['accuracy_before'] = compute_accuracy(logits, targets)

            #print("updating params...")
            self.model.zero_grad()
            params = plasticity_rules.custom_sgd(self.model,
                                                inner_loss,
                                                step_size=step_size,
                                                params=params,
                                                first_order=(not self.model.training) or first_order,
                                                custom_update_fn = self.custom_inner_update_fn)
            
            if self.inner_loop_quantizer is not None:
                params = quantize_parameters(params, self.inner_loop_quantizer)
            
            #print("updated parameters")
        return params, results

    def train(self, dataloader, max_batches=500, verbose=True, epoch=-1, **kwargs):
        with tqdm(total=max_batches, disable=False, **kwargs) as pbar:
            for results in self.train_iter(dataloader, max_batches=max_batches, epoch=epoch):
                pbar.update(1)
                postfix = {'loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                if 'accuracies_after' in results:
                    postfix['after'] = '{0:.4f}'.format(np.mean(results['accuracies_after']))
                if 'accuracies_before' in results:
                    postfix['before']  = '{0:.4f}'.format(np.mean(results['accuracies_before']))
                pbar.set_postfix(**postfix)

    #Outer loop
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
                #pdb.set_trace()
                outer_loss.backward()
                #pdb.set_trace()
                #self.model.grad_flow('./')
                #pdb.set_trace()
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

                if self.scheduler is not None:
                    self.scheduler.step()

                num_batches += 1

    def evaluate(self, dataloader, max_batches=500, verbose=True, **kwargs):
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        with tqdm(total=max_batches, disable=False, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss'] - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'accuracies_after' in results:
                    mean_accuracy += (np.mean(results['accuracies_after'])
                        - mean_accuracy) / count
                    postfix['after in-loop'] = '{0:.4f}'.format(np.mean(mean_accuracy))
                if 'accuracies_before' in results:
                    postfix['before in-loop']  = '{0:.4f}'.format(np.mean(results['accuracies_before']))
                pbar.set_postfix(**postfix)

        mean_results = {'mean_outer_loss': mean_outer_loss}
        if 'accuracies_after' in results:
            mean_results['accuracies_after'] = mean_accuracy

        return mean_results

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

MAML = ModelAgnosticMetaLearning

class FOMAML(ModelAgnosticMetaLearning):
    def __init__(self, model, optimizer=None, step_size=0.1,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=F.cross_entropy, device=None):
        super(FOMAML, self).__init__(model, optimizer=optimizer, first_order=True,
            step_size=step_size, learn_step_size=learn_step_size,
            per_param_step_size=per_param_step_size,
            num_adaptation_steps=num_adaptation_steps, scheduler=scheduler,
            loss_function=loss_function, device=device)
