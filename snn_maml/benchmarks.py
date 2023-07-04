import torch.nn.functional as F

from collections import namedtuple
from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision import transforms as trn #.transforms import ToTensor, Resize, Compose

from .utils import ToTensor1D
import torch

import pdb
import numpy as np


Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_val_dataset '
                                    'meta_test_dataset model loss_function input_size')

class Pad(object):
    from torch.nn import functional as F
    
    def __init__(self, padding=2):
        self.padding = padding
        
    def __call__(self, data):
        # zero pad an tpyx event array
        # this will allow adding to the dimensions without 
        #too negatively impacting data
        p4d = (1,0,0,1,0,0,0,0)
        return np.asarray(F.pad(torch.from_numpy(data), p4d, "constant", 0))


def get_benchmark_by_name(name,
                          folder,
                          num_ways,
                          num_ways_val,
                          num_shots,
                          num_shots_test,
                          detach_at=None,
                          hidden_size=None,
                          params_file=None,
                          device=None,
                          non_spiking=False):

    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)

    if name == 'omniglot':
        data_dir = folder + '/data'

        from .model import ModelConvOmniglot
        class_augmentations = [Rotation([90, 180, 270])]
        transform = trn.Compose([trn.Resize(28), trn.ToTensor()])
        size = [1, 28, 28]
        meta_train_dataset = Omniglot(data_dir,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=dataset_transform,
                                      download=True)
        meta_val_dataset = Omniglot(data_dir,
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Omniglot(data_dir,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)

        model = ModelConvOmniglot(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy


    elif name == 'doublenmnistsequence':

        from .snn_model import build_model_DECOLLE, build_model_DECOLLECUBA
        from torchneuromorphic.doublenmnist_torchmeta.doublenmnist_dataloaders import (DoubleNMNIST,
                                                                                       Compose,
                                                                                       ClassNMNISTDataset,
                                                                                       CropDims,
                                                                                       Downsample,
                                                                                       ToCountFrame,
                                                                                       ToTensor)

        data_dir = folder+'/data/nmnist/n_mnist.hdf5'

        transform = None
        target_transform = None
        
        remove_time_dim=False
        
        ds=2
        size = [2, 32//ds, 32//ds]
        
        if params_file is None:
            params_file = folder+'/parameters/decolle_params-CNN_REDECOLLE.yml'
            
        print("USING PARAMS FROM",params_file)
        
        with open(params_file, 'r') as f:
            import yaml
            params = yaml.safe_load(f)
            
        chunk_size = params['chunk_size_train']
        dt = params['deltat']
        transform = Compose([
            CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size, size = size),
            ToTensor()])

        if target_transform is None:
            target_transform = Categorical(num_ways)

        loss_function = F.cross_entropy
        
        # print("using num_ways_val")
        
        meta_train_dataset = ClassSplitter(DoubleNMNIST(root = data_dir,
                                                        meta_train=True,
                                                        transform = transform,
                                                        target_transform = target_transform,
                                                        chunk_size=chunk_size,
                                                        num_classes_per_task=num_ways), 
                                           num_train_per_class = num_shots, 
                                           num_test_per_class = num_shots_test)
        
        meta_val_dataset = ClassSplitter(DoubleNMNIST(root = data_dir,
                                                      meta_val=True,
                                                      transform = transform,
                                                      target_transform = target_transform,
                                                      chunk_size=chunk_size,
                                                      num_classes_per_task=num_ways_val),
                                         num_train_per_class = num_shots,
                                         num_test_per_class = num_shots_test)
        
        meta_test_dataset = ClassSplitter(DoubleNMNIST(root = data_dir,
                                                       meta_test=True,
                                                       transform = transform,
                                                       target_transform = target_transform,
                                                       chunk_size=chunk_size,
                                                       num_classes_per_task=num_ways_val), 
                                          num_train_per_class = num_shots, 
                                          num_test_per_class = num_shots_test)
        
        
        if 'cuba' in params.keys():
            if params['cuba']:
                model = build_model_DECOLLECUBA(num_ways, params_file = params_file, device=device, detach_at=detach_at,sg_function_baseline=non_spiking)
            else:
                model = build_model_DECOLLE(num_ways, params_file = params_file, device=device, detach_at=detach_at,sg_function_baseline=non_spiking)
        else:
            model = build_model_DECOLLE(num_ways, params_file = params_file, device=device, detach_at=detach_at,sg_function_baseline=non_spiking)
        
    elif name == 'doubledvssignsequence':
        from torchneuromorphic.double_dvssign.doubledvssign_dataloaders import DoubleDVSSign,Compose,ClassDVSSignDataset,CropDims,Downsample,ToCountFrame,ToTensor,ToEventSum,Repeat,toOneHot
        from torchneuromorphic.utils import plot_frames_imshow
        from matplotlib import pyplot as plt
        from torchmeta.utils.data import CombinationMetaDataset
        from .snn_model import build_model_DECOLLE, build_model_DECOLLECUBA

        #root = 'data/nmnist/n_mnist.hdf5'
        #Please do not hardcode, make use of the variable folder
        print("data is here", folder)
        root_dvssign = folder+'/data/ASL-DVS/dvssign.hdf5'
        print("DOUBLE ASL-DVS SPIKING")

        if params_file is None:
            params_file = folder+'/parameters/decolle_params-CNN-Sign.yml'
            
        print("USING PARAMS FROM",params_file)
        
        with open(params_file, 'r') as f:
            import yaml
            params = yaml.safe_load(f)

        chunk_size = 100
        ds = 6 # 60x30
        dt = 1000
        transform = None
        target_transform = None

        remove_time_dim = False

        size = [2, 240//ds, 180//ds]

        transform = Compose([
                CropDims(low_crop=[0,0], high_crop=[240,180], dims=[2,3]),
                Downsample(factor=[1000,1,ds,ds]),
                ToCountFrame(T = chunk_size, size = size),
                ToTensor()])

        if target_transform is None:
            target_transform = Compose([Repeat(chunk_size), toOneHot(num_ways)])

        loss_function = F.cross_entropy

        meta_split = folder + '/parameters/doubledvssign_splits_full.json'

        meta_train_dataset = ClassSplitter(DoubleDVSSign(root = root_dvssign, meta_train=True, meta_split=meta_split, transform = transform, target_transform = target_transform, chunk_size=chunk_size,  num_classes_per_task=num_ways), num_train_per_class = num_shots, num_test_per_class = num_shots_test)
        meta_val_dataset = ClassSplitter(DoubleDVSSign(root = root_dvssign, meta_val=True, meta_split=meta_split, transform = transform, target_transform = target_transform, chunk_size=chunk_size,  num_classes_per_task=num_ways), num_train_per_class = num_shots, num_test_per_class = num_shots_test)
        meta_test_dataset = ClassSplitter(DoubleDVSSign(root = root_dvssign, meta_test=True, meta_split=meta_split, transform = transform, target_transform = target_transform, chunk_size=chunk_size,  num_classes_per_task=num_ways), num_train_per_class = num_shots, num_test_per_class = num_shots_test)

        #model = build_model_DECOLLE(num_ways, params_file = params_file, device=device, detach_at=detach_at,sg_function_baseline=non_spiking)#,detach=detach)#hidden_size=hidden_size)
        
        if 'cuba' in params.keys():
            if params['cuba']:
                model = build_model_DECOLLECUBA(num_ways, params_file = params_file, device=device, detach_at=detach_at,sg_function_baseline=non_spiking)
            else:
                model = build_model_DECOLLE(num_ways, params_file = params_file, device=device, detach_at=detach_at,sg_function_baseline=non_spiking)


    elif 'nomniglotsequence' in name:
        from torchneuromorphic.nomniglot.nomniglot_dataloaders import NOmniglot,Compose,ClassNOmniglotDataset,CropDims,Downsample,ToCountFrame,ToTensor,ToEventSum,Repeat,toOneHot
        from torchneuromorphic.utils import plot_frames_imshow
        from matplotlib import pyplot as plt
        from torchmeta.utils.data import CombinationMetaDataset
        from .snn_model import build_model_DECOLLE

        pdb.set_trace()
        data_dir = folder+'/../data/nomniglot/nomniglot.hdf5'#folder+'/data/nomniglot/nomniglot.hdf5'
        print("NOmniglot "+data_dir)
        #params_file = '/home/kennetms/Documents/snn_maml/parameters/decolle_params-CNN-Sign.yml'
        print(params_file)
        chunk_size = 100
        ds = 10 #10
        dt = 100000
        transform = None
        target_transform = None

        remove_time_dim = False

        size = [2, 346//ds, 260//ds]

        transform = Compose([
            CropDims(low_crop=[0,0], high_crop=[346,260], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size, size = size),
            ToTensor()])

        if target_transform is None:
            target_transform = Categorical(num_ways)

        loss_function = F.cross_entropy
        
        meta_train_dataset = NOmniglot(root = data_dir,
                                                        meta_train=True,
                                                        transform = transform,
                                                        target_transform = target_transform,
                                                        chunk_size=chunk_size*dt,
                                                        num_classes_per_task=num_ways,
                                                        dataset_transform=dataset_transform)
        
        meta_val_dataset = NOmniglot(root = data_dir,
                                                      meta_val=True,
                                                      transform = transform,
                                                      target_transform = target_transform,
                                                      chunk_size=chunk_size*dt,
                                                      num_classes_per_task=num_ways,
                                                      dataset_transform=dataset_transform)

        
        meta_test_dataset = NOmniglot(root = data_dir,
                                                       meta_test=True,
                                                       transform = transform,
                                                       target_transform = target_transform,
                                                       chunk_size=chunk_size*dt,
                                                       num_classes_per_task=num_ways,
                                                       dataset_transform=dataset_transform)
        
        
        if 'lava' in name:
            from .snn_model_lava import build_model_lava
            import lava.lib.dl.slayer as slayer
            
            with open(params_file, 'r') as f:
                import yaml
                params = yaml.safe_load(f)
            
            if params['network']['analog_readout']:
                loss_function = F.cross_entropy #slayer.loss.SpikeMax(mode='logsoftmax').to(device) #F.cross_entropy #slayer.loss.SpikeMax(mode='logsoftmax') # SpikeMax functions like NLL for classification
            else:
                loss_function = slayer.loss.SpikeMax(mode='logsoftmax').to(device)
            
            model = build_model_lava(num_ways, params_file = params_file, device=device, detach_at=detach_at)
        else:
            loss_function = F.cross_entropy 
        
            model = build_model_DECOLLE(num_ways, params_file = params_file, device=device, detach_at=detach_at, sg_function_baseline=non_spiking)
        
        
        #model = build_model_DECOLLE(num_ways, params_file = params_file, device=device, detach_at=detach_at, sg_function_baseline=non_spiking)#,detach=detach)#hidden_size=hidden_size)
        
    elif name == 'doublenmnistlava':
        from .snn_model_lava import build_model_lava
        import lava.lib.dl.slayer as slayer
        from torchneuromorphic.doublenmnist_torchmeta.doublenmnist_dataloaders import (DoubleNMNIST,
                                                                                       Compose,
                                                                                       ClassNMNISTDataset,
                                                                                       CropDims,
                                                                                       Downsample,
                                                                                       ToCountFrame,
                                                                                       ToTensor)
        
        data_dir = folder+'/data/nmnist/n_mnist.hdf5'

        transform = None
        target_transform = None
        
        remove_time_dim=False
        
        ds=2
        size = [2, 32//ds, 32//ds]
        
        if params_file is None:
            params_file = folder+'/parameters/lava_params_dnmnist.yml'
            
        print("USING PARAMS FROM",params_file)
        
        with open(params_file, 'r') as f:
            import yaml
            params = yaml.safe_load(f)
            
        chunk_size = params['network']['chunk_size_train']
        dt = params['network']['deltat']
        transform = Compose([
            CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size, size = size),
            ToTensor()])

        if target_transform is None:
            target_transform = Categorical(num_ways)

        if params['network']['analog_readout']:
            loss_function = F.cross_entropy #slayer.loss.SpikeMax(mode='logsoftmax').to(device) #F.cross_entropy #slayer.loss.SpikeMax(mode='logsoftmax') # SpikeMax functions like NLL for classification
        else:
            loss_function = slayer.loss.SpikeMax(mode='logsoftmax').to(device)
        
        # print("using num_ways_val")
        
        meta_train_dataset = ClassSplitter(DoubleNMNIST(root = data_dir,
                                                        meta_train=True,
                                                        transform = transform,
                                                        target_transform = target_transform,
                                                        chunk_size=chunk_size,
                                                        num_classes_per_task=num_ways), 
                                           num_train_per_class = num_shots, 
                                           num_test_per_class = num_shots_test)
        
        meta_val_dataset = ClassSplitter(DoubleNMNIST(root = data_dir,
                                                      meta_val=True,
                                                      transform = transform,
                                                      target_transform = target_transform,
                                                      chunk_size=chunk_size,
                                                      num_classes_per_task=num_ways_val),
                                         num_train_per_class = num_shots,
                                         num_test_per_class = num_shots_test)
        
        meta_test_dataset = ClassSplitter(DoubleNMNIST(root = data_dir,
                                                       meta_test=True,
                                                       transform = transform,
                                                       target_transform = target_transform,
                                                       chunk_size=chunk_size,
                                                       num_classes_per_task=num_ways_val), 
                                          num_train_per_class = num_shots, 
                                          num_test_per_class = num_shots_test)
        
        model = build_model_lava(num_ways, params_file = params_file, device=device, detach_at=detach_at)
        
        
    elif name == 'doubledvssignlava':
        from .snn_model_lava import build_model_lava
        import lava.lib.dl.slayer as slayer
        from torchneuromorphic.double_dvssign.doubledvssign_dataloaders import DoubleDVSSign,Compose,ClassDVSSignDataset,CropDims,Downsample,ToCountFrame,ToTensor,ToEventSum,Repeat,toOneHot
        
        
        print("data is here", folder)
        data_dir = folder+'/data/ASL-DVS/dvssign.hdf5'
        print("DOUBLE ASL-DVS SPIKING Lava")
        
        print("USING PARAMS FROM",params_file)
        
        if params_file is None:
            print("NEED PARAMS FILE!")
            1/0
        
        with open(params_file, 'r') as f:
            import yaml
            params = yaml.safe_load(f)

        chunk_size = params['network']['chunk_size_train']
        ds = 12 #6
        dt = 1000
        transform = None
        target_transform = None

        remove_time_dim = False

        size = [2, 240//16, 180//ds]

        transform = Compose([
                CropDims(low_crop=[0,0], high_crop=[240,180], dims=[2,3]),
                Downsample(factor=[1000,1,16,ds]),
                ToCountFrame(T = chunk_size, size = size),
                Pad(),
                ToTensor()])

        if target_transform is None:
            target_transform = Categorical(num_ways)

        if params['network']['analog_readout']:
            loss_function = F.cross_entropy #slayer.loss.SpikeMax(mode='logsoftmax').to(device) #F.cross_entropy #slayer.loss.SpikeMax(mode='logsoftmax') # SpikeMax functions like NLL for classification
        else:
            loss_function = slayer.loss.SpikeMax(mode='logsoftmax').to(device)
        
        # print("using num_ways_val")
        
        meta_train_dataset = ClassSplitter(DoubleDVSSign(root = data_dir,
                                                        meta_train=True,
                                                        transform = transform,
                                                        target_transform = target_transform,
                                                        chunk_size=chunk_size,
                                                        num_classes_per_task=num_ways), 
                                           num_train_per_class = num_shots, 
                                           num_test_per_class = num_shots_test)
        
        meta_val_dataset = ClassSplitter(DoubleDVSSign(root = data_dir,
                                                      meta_val=True,
                                                      transform = transform,
                                                      target_transform = target_transform,
                                                      chunk_size=chunk_size,
                                                      num_classes_per_task=num_ways_val),
                                         num_train_per_class = num_shots,
                                         num_test_per_class = num_shots_test)
        
        meta_test_dataset = ClassSplitter(DoubleDVSSign(root = data_dir,
                                                       meta_test=True,
                                                       transform = transform,
                                                       target_transform = target_transform,
                                                       chunk_size=chunk_size,
                                                       num_classes_per_task=num_ways_val), 
                                          num_train_per_class = num_shots, 
                                          num_test_per_class = num_shots_test)
        
        model = build_model_lava(num_ways, params_file = params_file, device=device, detach_at=detach_at)
        
        
    elif 'dvsgesturemeta' in name:
        from torchneuromorphic.dvs_gestures_torchmeta.dvsgestures_dataloaders_meta import (DVSGestureMeta,
                                                                                       Compose,
                                                                                       ClassDVSGestureMetaDataset,
                                                                                       CropDims,
                                                                                       Downsample,
                                                                                       ToCountFrame,
                                                                                       ToTensor)
        from torchneuromorphic.utils import plot_frames_imshow
        from matplotlib import pyplot as plt
        from torchmeta.utils.data import CombinationMetaDataset
        from .snn_model import build_model_DECOLLE
        import lava.lib.dl.slayer as slayer

        data_dir = folder+'/../data/dvs_gesture_meta.hdf5'#folder+'/data/nomniglot/nomniglot.hdf5'
        print("DVSGesture Meta "+data_dir)
        #params_file = '/home/kennetms/Documents/snn_maml/parameters/decolle_params-CNN-Sign.yml'
        print(params_file)
        #chunk_size = 100
        
        with open(params_file, 'r') as f:
            import yaml
            params = yaml.safe_load(f)
            
        if 'lava' in name:
            chunk_size = params['network']['chunk_size_train']
        else:
            chunk_size = params['chunk_size_train']
        
        ds = 4
        dt = 1000
        transform = None
        target_transform = None

        remove_time_dim = False

        size = [2, 128//ds, 128//ds]

        transform = Compose([
            CropDims(low_crop=[0,0], high_crop=[128,128], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size, size = size),
            ToTensor()])

        if target_transform is None:
            target_transform = Categorical(num_ways)

        
        
        meta_train_dataset = DVSGestureMeta(root = data_dir,
                                                        meta_train=True,
                                                        transform = transform,
                                                        target_transform = target_transform,
                                                        chunk_size=chunk_size*dt,
                                                        num_classes_per_task=num_ways,
                                                        dataset_transform=dataset_transform)
        
        meta_val_dataset = DVSGestureMeta(root = data_dir,
                                                       meta_test=True,
                                                       transform = transform,
                                                       target_transform = target_transform,
                                                       chunk_size=chunk_size*dt,
                                                       num_classes_per_task=num_ways,
                                                       dataset_transform=dataset_transform)

        
        meta_test_dataset = DVSGestureMeta(root = data_dir,
                                                       meta_test=True,
                                                       transform = transform,
                                                       target_transform = target_transform,
                                                       chunk_size=chunk_size*dt,
                                                       num_classes_per_task=num_ways,
                                                       dataset_transform=dataset_transform)
        
        if 'lava' in name:
            from .snn_model_lava import build_model_lava
            import lava.lib.dl.slayer as slayer
            
            if params['network']['analog_readout']:
                loss_function = F.cross_entropy #slayer.loss.SpikeMax(mode='logsoftmax').to(device) #F.cross_entropy #slayer.loss.SpikeMax(mode='logsoftmax') # SpikeMax functions like NLL for classification
            else:
                loss_function = slayer.loss.SpikeMax(mode='logsoftmax').to(device)
            
            model = build_model_lava(num_ways, params_file = params_file, device=device, detach_at=detach_at)
        else:
            loss_function = F.cross_entropy 
        
            model = build_model_DECOLLE(num_ways, params_file = params_file, device=device, detach_at=detach_at, sg_function_baseline=non_spiking)#,detach=
            
        
    elif 'emg' in name:
        from torchneuromorphic.emg_meta.emg_dataloaders_meta import (EMGMeta,
                                                                                   Compose,
                                                                                   ClassEMGMetaDataset,
                                                                                   CropDims,
                                                                                   Downsample,
                                                                                   ToCountFrame,
                                                                                   ToTensor)
        from torchneuromorphic.utils import plot_frames_imshow
        from matplotlib import pyplot as plt
        from torchmeta.utils.data import CombinationMetaDataset
        from .snn_model import build_model_DECOLLE
        import lava.lib.dl.slayer as slayer

        data_dir = folder+'/../data/emg_meta.hdf5'#folder+'/data/nomniglot/nomniglot.hdf5'
        print("EMG Meta "+data_dir)
        #params_file = '/home/kennetms/Documents/snn_maml/parameters/decolle_params-CNN-Sign.yml'
        print(params_file)


        with open(params_file, 'r') as f:
            import yaml
            params = yaml.safe_load(f)

        
        size = params['input_shape']
        
        target_transform = Categorical(num_ways)


        meta_train_dataset = EMGMeta(root = data_dir,
                                                    meta_train=True,
                                                    num_classes_per_task=num_ways,
                                                    dataset_transform=dataset_transform)

        meta_val_dataset = EMGMeta(root = data_dir,
                                                   meta_test=True,
                                                   num_classes_per_task=num_ways,
                                                   dataset_transform=dataset_transform)


        meta_test_dataset = EMGMeta(root = data_dir,
                                                   meta_test=True,
                                                   num_classes_per_task=num_ways,
                                                   dataset_transform=dataset_transform)

            
        
        if 'lava' in name:
            from .snn_model_lava import build_model_lava
            import lava.lib.dl.slayer as slayer
            
            if params['network']['analog_readout']:
                loss_function = F.cross_entropy #slayer.loss.SpikeMax(mode='logsoftmax').to(device) #F.cross_entropy #slayer.loss.SpikeMax(mode='logsoftmax') # SpikeMax functions like NLL for classification
            else:
                loss_function = slayer.loss.SpikeMax(mode='logsoftmax').to(device)
            
            model = build_model_lava(num_ways, params_file = params_file, device=device, detach_at=detach_at)
        else:
            loss_function = F.cross_entropy 
        
            model = build_model_DECOLLE(num_ways, params_file = params_file, device=device, detach_at=detach_at, sg_function_baseline=non_spiking)#,detach=detach)#hidden_size=hidden_size)
        
        
        
        
        
    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function,
                     input_size = size)
                                                  
