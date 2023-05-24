import torch
import math
import os
import time
import json
import logging
import numpy as np
import warnings
import wandb

from torchmeta.utils.data import BatchMetaDataLoader

from snn_maml.utils import tensors_to_device, compute_accuracy

from snn_maml.benchmarks import get_benchmark_by_name
import snn_maml.utils as utils

import argparse

parser = argparse.ArgumentParser('MAML')

# General
parser.add_argument('--output-folder', default = './logs_tmp', type=str, help='Path to the output folder to save the model.') 
parser.add_argument('--benchmark', type=str, default='doublenmnistsequence', help='Name of the dataset (default: omniglot).')
parser.add_argument('--folder', type=str, default='./', help='Root path containing parameters/ and data/.')
parser.add_argument('--num-ways', type=int, default=5, help='Number of classes per task (N in "N-way", default: 5).')
parser.add_argument('--num-shots', type=int, default=1, help='Number of training example per class (k in "k-shot", default: 5).')
parser.add_argument('--num-shots-test', type=int, default=10, help='Number of test example per class. If negative, same as the number of training examples `--num-shots` (default: 15).')
parser.add_argument('--warm-start', type=str, default='', help='model file to load for warm start')
parser.add_argument('--boil', action='store_true', help='body only in inner loop')
parser.add_argument('--quantize', type=str, default=None, help='quantize weights')
parser.add_argument('--quantize_in', type=str, default=None, help='quantize weights')
parser.add_argument('--learn-step-size', action='store_true', help='Learn step sizes')
parser.add_argument('--per-param-step-size', action='store_true', help='Learn per module step size')

# Model
parser.add_argument('--hidden-size', type=int, default=64,
    help='Number of channels in each convolution layer of the VGG network '
    '(default: 64).')

parser.add_argument('--burnin', type=int, default=70, help='Steps to "burnin" (default: 70).')

parser.add_argument('--load-model',type=str, default='', help='Path to the model file to load (default: "")')

# Optimization
parser.add_argument('--batch-size', type=int, default=5, help='Number of tasks in a batch of tasks (default: 5).')
parser.add_argument('--num-steps', type=int, default=1, help='Number of fast adaptation steps, ie. gradient descent '
    'updates (default: 1).')
parser.add_argument('--num-epochs', type=int, default=200, help='Number of epochs of meta-training (default: 50).')
parser.add_argument('--num-batches', type=int, default=100, help='Number of batch of tasks per epoch (default: 100).')
parser.add_argument('--num-batches-test', type=int, default=100, help='Number of batch of tasks per epoch (default: 100).')
parser.add_argument('--step-size', type=float, default=1.0, help='Size of the fast adaptation step, ie. learning rate in the gradient descent update (default: 1.0).')
parser.add_argument('--soel-threshold', type=float, default=.05, help='Threshold applied to the SOEL update')
parser.add_argument('--first-order', action='store_true', help='Use the first order approximation, do not use higher-order derivatives during meta-optimization.')
parser.add_argument('--nonspiking', action='store_true', help='Activate baseline for non-surrogate (non-spiking)')
parser.add_argument('--meta-lr', type=float, default=1e-3,  help='Learning rate for the meta-optimizer (optimization of the outer loss). The default optimizer is Adam (default: 2e-3).')

# Misc
parser.add_argument('--num-workers', type=int, default=8, help='Number of workers to use for data-loading (default: 4).')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('--device-nonlin', default=None, type=str, help='Use device non-linearity which wraps the gradient update provided as import path (default=None)')
parser.add_argument('--weight-clamp', default=None, type=str, help='Clamp the weight using a hard-coded function provided as import path (default=None)')
parser.add_argument('--params_file', type=str, default='parameters/decolle_params-CNN.yml', help='Path to the parameters file if dcll is used.')
parser.add_argument('--metalearner', type=str, default='MAML', help='Metalearner to use (detaul: "MAML", other options: "SOEL")')
parser.add_argument('--do-test', action='store_true')
parser.add_argument('--do-train', action='store_true')
parser.add_argument('--deltaw', type=float, default=None, help='Force larger weight changes. The larger the value the larger the deltaw needs to be for params to update. (default None)')
parser.add_argument('--detach-at', type=int, default=None, help='Detach part of the network from specified layer (default None).')
parser.add_argument('--device', type=int, default=0, help='Which gpu to use if multiple available (default 0).')
parser.add_argument('--no-log', action='store_true')


# parser.add_argument('--deltaw', type=float, default=None, help='Force larger weight changes. The larger the value the larger the deltaw needs to be for params to update. (default None)')

args = parser.parse_args()


if args.metalearner == 'MAML':
    from snn_maml.maml import ModelAgnosticMetaLearning as metalearner_model
    from importlib import import_module
    dev_nonlin_fun = getattr(custom_funs, args.device_nonlin) if args.device_nonlin is not None else None
    dev_clamp = getattr(custom_funs, args.weight_clamp) if args.weight_clamp is not None else None
    add_kwargs = {'custom_inner_update_fn': dev_nonlin_fun,
                  'custom_outer_update_fn': dev_clamp}
    
elif args.metalearner == 'SOEL':
    from snn_maml.maml_with_soel import ModelAgnosticMetaLearning_With_SOEL as metalearner_model
    add_kwargs = {'threshold':args.soel_threshold}

if not args.do_train and not args.do_test:
    args.do_train = True # default to training

if args.num_shots_test <= 0:
    args.num_shots_test = args.num_shots

logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
device = torch.device(f'cuda:{args.device}' if not args.no_cuda and torch.cuda.is_available() else 'cpu')

if (args.output_folder is not None):
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logging.debug('Creating folder `{0}`'.format(args.output_folder))

    output_folder = os.path.join(args.output_folder,
                          time.strftime('%Y-%m-%d_%H%M%S'))
    os.makedirs(output_folder)
    args.output_folder=output_folder
    logging.debug('Creating folder `{0}`'.format(output_folder))

    args.folder = os.path.abspath(args.folder)
    args.model_path = os.path.abspath(os.path.join(output_folder, 'model.th'))
    args.opt_path = os.path.abspath(os.path.join(output_folder, 'optim.th'))
    args.stepsize_path = os.path.abspath(os.path.join(output_folder, 'stepsize.th'))
    # Save the configuration in a config.json file
    json_file_path = os.path.join(output_folder, 'config.json')
    with open(json_file_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logging.info('Saving configuration file in `{0}`'.format(
                 os.path.abspath(json_file_path)))

    if not args.no_log: wandb.login()
    with open(json_file_path, 'r') as f:
        print(json_file_path)
        if not args.no_log: wandb.init(project = "snn_maml", name = f"{output_folder}", config=json.load(f))

benchmark = get_benchmark_by_name(args.benchmark,
                                  args.folder,
                                  args.num_ways,
                                  args.num_ways, #validation
                                  args.num_shots,
                                  args.num_shots_test,
                                  detach_at=args.detach_at,
                                  hidden_size=args.hidden_size,
                                  params_file = args.params_file,
                                  device=device,
                                  non_spiking = args.nonspiking)
net = benchmark.model

meta_train_dataloader = BatchMetaDataLoader(benchmark.meta_train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            pin_memory=True)

meta_val_dataloader = BatchMetaDataLoader(benchmark.meta_val_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers,
                                          pin_memory=True)

if args.do_test:
    meta_test_dataloader = BatchMetaDataLoader(benchmark.meta_test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)



if hasattr(benchmark.model, 'get_trainable_parameters'):
    print('Using get_trainable_parameters instead of parameters for optimization parameters')
    meta_optimizer = torch.optim.Adam(benchmark.model.parameters(), lr=args.meta_lr) 
    if args.quantize :
        print('Quantize')
        from snn_maml.utils import create_fixed_quantizers
        quantizer_out = create_fixed_quantizers()[args.quantize]
    else:
        quantizer_out = None
        
    if args.quantize_in :
        print('Quantize')
        from snn_maml.utils import create_fixed_quantizers
        quantizer_in = create_fixed_quantizers()[args.quantize_in]
        
    else:
        quantizer_in = None

    meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer,
                                                                eta_min=args.meta_lr/50,
                                                                T_max=args.num_epochs)
else:
    quantizer_out = None
    quantizer_in = None
    meta_optimizer = torch.optim.Adam(benchmark.model.parameters(), lr=args.meta_lr) 
    meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer,
                                                                eta_min=args.meta_lr/50,
                                                                T_max=args.num_epochs)
    


    
print(benchmark.model)

metalearner = metalearner_model(benchmark.model,
                                        meta_optimizer,
                                        first_order=args.first_order,
                                        num_adaptation_steps=args.num_steps,
                                        step_size=args.step_size,
                                        learn_step_size=args.learn_step_size,
                                        loss_function=benchmark.loss_function,
                                        scheduler=meta_scheduler,
                                        device=device,
                                        per_param_step_size=args.learn_step_size and args.per_param_step_size,
                                        boil = args.boil,
                                        outer_loop_quantizer = quantizer_out,
                                        inner_loop_quantizer = quantizer_in,
                                        **add_kwargs)

    
print("Using metalearner ",metalearner)
best_value = None

out = next(iter(meta_train_dataloader))
out_c = tensors_to_device(out, device=device)



if args.warm_start != '':
    net.load_state_dict(torch.load(args.warm_start+'model.th'))
    #try:
    #    metalearner.optimizer.load_state_dict(torch.load(args.warm_start+'optim.th'))
    #except:
    #    warnings.warn('No optimization loading')
    try:
        step_size = torch.load(args.warm_start+'stepsize.th')
        metalearner.step_size.data = step_size
    except FileNotFoundError:
        warnings.warn('No step size loading')

elif hasattr(net, 'LIF_layers'):
    if args.params_file is not None:
        with open(args.params_file, 'r') as f:
            import yaml
            params = yaml.load(f, Loader=yaml.FullLoader)
        if args.burnin != -1:
            params['burnin_steps'] = args.burnin
            net.burnin = params['burnin_steps']
    else:
        raise Exception('Must provide params_file')

    from decolle.init_functions import init_LSUV_actrate
    dd=out_c['train'][0].reshape(-1,params['chunk_size_train'],*params['input_shape'])
    if not args.nonspiking:
        init_LSUV_actrate(net, dd, params['act_rate']) #0.288 is hard coded from params['actrate']
    
if args.load_model:
    print("loading model")
    with open(args.load_model, 'rb') as f:
        benchmark.model.load_state_dict(torch.load(f, map_location=device))
        print(benchmark.model)
        if os.path.exists(args.load_model[:-8]+'optim.th'):
            metalearner.optimizer.load_state_dict(torch.load(args.load_model[:-8]+'optim.th')) # -8 because we need to remove model.th

epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
results_accuracy_after = []

all_test = np.zeros(args.num_epochs)
all_train = np.zeros(args.num_epochs)


for epoch in range(args.num_epochs):
    print(epoch, meta_scheduler.get_last_lr())
    if args.do_train:
        results_train = metalearner.train(meta_train_dataloader,
                          max_batches=args.num_batches,
                          verbose=args.verbose,
                          desc='Training',
                          leave=False,
                          epoch = epoch)#,
                          #deltaw=args.deltaw)
        
        if results_train is not None:
            all_train[epoch] = np.mean(results_train['accuracies_after'])
    results = metalearner.evaluate(meta_val_dataloader,
                                   max_batches=args.num_batches_test,
                                   verbose=args.verbose,
                                   desc=epoch_desc.format(epoch + 1))


    if 'accuracies_after' in results:
        results_accuracy_after.append(results['accuracies_after'])
        if not args.no_log: wandb.log({'accuracies_after/':results['accuracies_after'], 'epoch':epoch}) 
        np.save(args.output_folder+'test_acc.npy',results_accuracy_after)
        save_model =True

        if save_model and (args.output_folder is not None):
            with open(args.model_path, 'wb') as f:
                torch.save(benchmark.model.state_dict(), f)
            with open(args.opt_path, 'wb') as f:
                torch.save(metalearner.optimizer.state_dict(), f)
            with open(args.stepsize_path, 'wb') as f:
                torch.save(metalearner.step_size, f)
                
    if args.do_test:
        results_test = metalearner.evaluate(meta_test_dataloader,
                           max_batches=args.num_batches_test,
                           verbose=args.verbose,
                           desc=epoch_desc.format(epoch + 1))#,
                           #deltaw=args.deltaw)
        
        print("Test results: ", np.mean(results_test['accuracies_after']))

        all_test[epoch] = np.mean(results_test['accuracies_after'])
                
if args.do_train:
    print("mean train", np.mean(all_train))
    print("stddev train", np.std(all_train))
    
    if (best_value is None) or (best_value < results['accuracies_after']):
        best_value = results['accuracies_after']
        save_model = True
    elif (best_value is None) or (best_value > results['mean_outer_loss']):
        best_value = results['mean_outer_loss']
        save_model = True
    else:
        save_model = False
        
    if save_model and (args.output_folder is not None):
        with open(args.model_path, 'wb') as f:
            torch.save(benchmark.model.state_dict(), f)
        with open(args.opt_path, 'wb') as f:
            torch.save(metalearner.optimizer.state_dict(), f)
        with open(args.stepsize_path, 'wb') as f:
                torch.save(metalearner.step_size, f)

if args.do_test:
    print("mean test", np.mean(all_test))
    print("stddev test", np.std(all_test))

if hasattr(benchmark.meta_train_dataset, 'close'):
    benchmark.meta_train_dataset.close()
    benchmark.meta_val_dataset.close()
