import sys
import os 
sys.path.append(os.getcwd())
import torch
from pathlib import Path
import logging
import timm
from pactl.nn import create_model
from pactl.nn.projectors import create_intrinsic_model
from pactl.data import get_dataset
from torch.utils.data import DataLoader
from pactl.random import random_seed_all
from tqdm.auto import tqdm
from pactl.bounds.quantize_fns import Quantize


from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pactl.distributed import maybe_launch_distributed
from pactl.logging import set_logging, finish_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset
from pactl.nn import create_model
from pactl.bounds.get_bound_from_chk_v2 import evaluate_idmodel


def load_saved_model(prenet_cfg_path,model_class, model_path,device_id = 0,log_dir=None):
    device=torch.device(f"cuda:{device_id}")
    #model = create_intrinsic_model(base_net = create_model(model_name = model_class,device_id =      device_id),intrinsic_dim = 7500,intrinsic_mode = "filmrdkron",seed = 137)

    model = create_model(cfg_path=prenet_cfg_path,
                       device_id=device_id,
                       log_dir=log_dir)
    
    model = model.to(device)
    model.subspace_params.data = torch.load(model_path).to(device)
    return model


def compute_accuracy(model, loader, device,train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()
        
    correct = 0
    total = 0
    #quantize = Quantize.apply
  
    for inputs, labels in tqdm(loader, desc='Computing accuracy'):
        inputs, labels = inputs.to(device), labels.to(device)
            
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
            
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100  
    return accuracy
    
def main(prenet_cfg_path,model_class, model_path,dataset,batch_size=256,seed = 137,data_dir=None,train_subset=1.,
    indices_path=None, num_workers=4,device_id = 0,log_dir = None,distributed = False):
    
    random_seed_all(seed)
    device=torch.device(f"cuda:{device_id}")
    
    model = load_saved_model(prenet_cfg_path,model_class, model_path,log_dir = log_dir)
    
    train_data, test_data = get_dataset(dataset,
                                        root=data_dir,
                                        train_subset=train_subset,
                                        indices_path=indices_path)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        sampler= None)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler= None)
    
    train_acc = compute_accuracy(model,train_loader,device,train_mode = True)
    print("train accuracy on train mode: ",train_acc)
    train_acc = compute_accuracy(model,train_loader,device,train_mode = False)
    print("train accuracy on eval mode: ",train_acc)
    test_acc = compute_accuracy(model,test_loader,device,train_mode = False)
    print("test accuracy on eval mode: ",test_acc)

    
    train_acc = compute_accuracy(model._forward_net[0],train_loader,device,train_mode = False)
    print("forward net train accuracy on eval mode: ",train_acc)
    test_acc = compute_accuracy(model._forward_net[0],test_loader,device,train_mode = False)
    print("forward net test accuracy on eval mode: ",test_acc)

   

def entrypoint(log_dir=None, **kwargs):
    world_size, rank, device_id = maybe_launch_distributed()

    if 'device_id' in list(kwargs.keys()):
        device_id = kwargs['device_id']
        kwargs.pop('device_id')
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(device_id)

    ## Only setup logging from one process (rank = 0).
    log_dir = set_logging(log_dir=log_dir) if rank == 0 else None
    if rank == 0:
        logging.info(f'Working with {world_size} process(es).')

    main(**kwargs,
         log_dir=log_dir,
         distributed=(world_size > 1),
         device_id=device_id)

    if rank == 0:
        finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)

