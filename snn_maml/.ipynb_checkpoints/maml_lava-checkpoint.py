import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from snn_maml.utils import quantize_parameters
from collections import OrderedDict
from . import plasticity_rules
from .utils import tensors_to_device, compute_accuracy, compute_accuracy_lava
from .maml import ModelAgnosticMetaLearning

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


class ModelAgnosticMetaLearning_Lava(ModelAgnosticMetaLearning):
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
                 inner_loop_quantizer = None,
                 use_soel=False):
        
        self.threshold = torch.tensor([.05], requires_grad=True, dtype=torch.float).to(device)
        print("Using quantiziation, delay, and spike rates with compute_accuracy_lava")
        
        self.use_soel = use_soel

        super(ModelAgnosticMetaLearning_Lava, self).__init__(
            model=model, 
            optimizer=optimizer,
            step_size=step_size,
            first_order=first_order,
            learn_step_size=learn_step_size,
            per_param_step_size=per_param_step_size,
            num_adaptation_steps=num_adaptation_steps,
            scheduler=scheduler,
            loss_function=loss_function,
            custom_outer_update_fn = None, custom_inner_update_fn = None,
            device=device, 
            boil=boil,
            outer_loop_quantizer = outer_loop_quantizer,
            inner_loop_quantizer = inner_loop_quantizer,
            )
        
        
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
        
            time_train_targets = torch.zeros((train_targets.shape[0],train_inputs.shape[0],100),dtype=torch.long).to(self.device)
            time_test_targets = torch.zeros((test_targets.shape[0],test_inputs.shape[0],100),dtype=torch.long).to(self.device)
            for i in range(train_targets.shape[0]):
                time_train_targets[i][train_targets[i]] = torch.ones(100)
            for i in range(test_targets.shape[0]):
                time_test_targets[i][test_targets[i]] = torch.ones(100)
                
                
            params, adaptation_results = self.adapt( 
                train_inputs, train_targets, time_train_targets,
                is_classification_task=is_classification_task,
                num_adaptation_steps=self.num_adaptation_steps,
                step_size=self.step_size, first_order=self.first_order)

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            
            if is_classification_task:
                results['accuracies_before'][task_id] = adaptation_results['accuracy_before']

                
            #pdb.set_trace()
            #print("test phase")
            with torch.set_grad_enabled(self.model.training):
                test_logits = self.model(test_inputs, params=params)
                
                if not self.use_soel:
                    outer_loss = self.loss_function(test_logits, test_targets)
                else:
                    outer_loss = self.loss_function(test_logits[:,:,-1], test_targets)    
                    
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

            if is_classification_task:
                results['accuracies_after'][task_id] = compute_accuracy_lava(
                    test_logits, test_targets)


        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()
        
        #i=1/0

        return mean_outer_loss, results
        
   
    #Inner loop
    def adapt(self, inputs, targets, time_targets, is_classification_task=None,
              num_adaptation_steps=1, step_size=0.1, first_order=False):
        if is_classification_task is None:
            is_classification_task = (not targets.dtype.is_floating_point)
            
        # pdb.set_trace()
        params = OrderedDict(self.model.meta_named_parameters())
        
        if self.outer_loop_quantizer is not None:
            params = quantize_parameters(params, self.outer_loop_quantizer)

        results = {'inner_losses': np.zeros(
            (num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):
            #pdb.set_trace()
            #print("getting adapt logits")
            logits = self.model(inputs, params=params)

            inner_loss = self.loss_function(logits, targets) #[:,:,-1], targets)
            results['inner_losses'][step] = inner_loss.item()
            #pdb.set_trace()
            if (step == num_adaptation_steps-1) and is_classification_task: 
                results['accuracy_before'] = compute_accuracy_lava(logits, targets)

            #print("updating params...")
            self.model.zero_grad()
            
            # pdb.set_trace()
            # print('before')
            
            if not self.use_soel:
            
                params = plasticity_rules.custom_sgd(self.model,
                                                    inner_loss,
                                                    step_size=step_size,
                                                    params=params,
                                                    first_order=(not self.model.training) or first_order,
                                                    custom_update_fn = self.custom_inner_update_fn)
            
            # logits = torch.mean(logits,dim=-1)
            else:
                pdb.set_trace()
                params = plasticity_rules.maml_soel(self.model,
                                                    logits,
                                                    time_targets,#batch_one_hot(time_targets,5).to(self.device),
                                                    step_size=step_size,
                                                    params=params,
                                                    first_order=(not self.model.training) or first_order,
                                                    threshold = self.threshold)
            
            # pdb.set_trace()
            # print('after')
            
            if self.inner_loop_quantizer is not None:
                for i,k in enumerate(params.keys()):
                    #pdb.set_trace()
                    params[k] = self.model.blocks[i].synapse._pre_hook_fx(params[k],descale=True)
                # pdb.set_trace()
                # params = quantize_parameters(params, self.inner_loop_quantizer)
            
            #print("updated parameters")
        return params, results
