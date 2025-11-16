import sys
import os 
sys.path.append(os.getcwd())
import logging
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, RMSprop, AdamW
from pactl.distributed import maybe_launch_distributed
import numpy as np
import pandas as pd
import json


from pactl.logging import set_logging, wandb, finish_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset

from ELLA_Subspace.meta_learner import Meta_ELLA
from ELLA_Subspace.projectors import unflatten_like
from data.Data_gen import data_gen

def main(seed=137, device_id=0, distributed=False, data_dir=None, log_dir=None,
        train_subset=1, indices_path=None, label_noise=0, num_workers=2,
         cfg_path=None, transfer=False, model_name='FcNet3', base_width=8,
         batch_size=64, optimizer='adam', lr=1e-2, momentum=.9, weight_decay=5e-4, epochs=2, global_levels=20, all_train_task_levels = 20,
         intrinsic_dim=600, intrinsic_mode='filmrdkron',is_said = False,is_layered = False, warmup_epochs=0, warmup_lr=.1, 
         levels = 5, use_kmeans = True, misc_extra_bits = 7, quant_epochs = 30, quant_lr = 0.0001, quantize_type = 'default',dataset='Omniglot', N_Way=5 ,K_Shot_MetaTrain=15, K_Shot_MetaTest=15,
         basis_num = 5, data_seed=42, data_transform = 'Shuffled_Pixels',n_train_tasks=10, n_test_tasks=20, samples_per_train=600, samples_per_test=600):

    
    
    train_loader, test_loader = data_gen(seed=data_seed, dataset=dataset, num_classes=N_Way, K_Shot_MetaTrain=K_Shot_MetaTrain, K_Shot_MetaTest=K_Shot_MetaTest, data_transform = data_transform, n_train_tasks=n_train_tasks, n_test_tasks = n_test_tasks, samples_per_train=samples_per_train, samples_per_test=samples_per_test)
    random_seed_all(seed)

    criterion = nn.CrossEntropyLoss()

    meta_learner = Meta_ELLA(train_loader, test_loader, intrinsic_dim, basis_num = basis_num, model_name=model_name, base_width=base_width,
                   seed=seed, intrinsic_mode=intrinsic_mode,cfg_path=cfg_path, transfer=transfer, device_id=device_id, log_dir=log_dir,
                   levels = levels, use_kmeans = use_kmeans, misc_extra_bits = misc_extra_bits, quant_epochs = quant_epochs, quant_lr = quant_lr,
                   quantize_type = quantize_type, global_levels=global_levels, all_train_task_levels = all_train_task_levels)
   
    filename = 'ELLA_Subspace/ELLA_global_params/meta_learner_omniglot_after_quant.txt'#+str(global_levels)+'.txt'
    with open(filename, 'r') as f:
        loaded_dict = json.load(f)

    
    parameters = {key: np.array(value) for key, value in loaded_dict.items()}

  
    meta_learner.global_param.data = torch.from_numpy(parameters['global_param']).float().to(device_id)

    for i in range(meta_learner.num_train_tasks):
        meta_learner.nets[i].subspace_params.data = torch.from_numpy(parameters[f'local_param_{i}']).float().to(device_id)
    print(meta_learner.nets[0].global_param)
    print("Parameters successfully reloaded into the model.")
    print(meta_learner.global_param.requires_grad)
    print(meta_learner.global_param[:20])
    sum_train_acc = 0
    sum_test_acc = 0
    for t in range(meta_learner.num_train_tasks):
      train_acc = meta_learner.meta_eval(meta_learner.nets[t], meta_learner.train_loader[t]['train'], criterion, device_id=device_id)['acc']
      sum_train_acc += train_acc
      test_acc = meta_learner.meta_eval(meta_learner.nets[t], meta_learner.train_loader[t]['test'], criterion, device_id=device_id)['acc']
      sum_test_acc  += test_acc
    
    
    print("average train accuracy on train data : ",sum_train_acc/meta_learner.num_train_tasks)
    print("average test accuracy on train data: ",sum_test_acc/meta_learner.num_train_tasks)
    print(meta_learner.global_param[:20])
    return meta_learner

#['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias', 'fc_out.weight', 'fc_out.bias']
def create_layered_params(meta_learner):    
    layer_params = []
    for i in range(meta_learner.basis_num):
        a = torch.zeros(meta_learner.basis_num)
        a[i] = 1
        coefficients = torch.repeat_interleave(a, meta_learner.d).to(meta_learner.global_param.device)
        weights = coefficients * meta_learner.global_param
        flat_projected_params = meta_learner.global_P @ weights
    
        unflattened_params = unflatten_like(
        flat_projected_params, meta_learner.nets[0].trainable_initparams)
        layer_params.append(unflattened_params)
    
    return layer_params
    
def calculate_norm(layer_params):
    basis_num = len(layer_params)
    layer_num = len(layer_params[0])

    l2_norms = []
    for l in range(layer_num):
        vectors = [layer_params[i][l] for i in range(basis_num)]
        norm = {} 
        for i in range(basis_num):
            for j in range(i + 1, basis_num):   
                l2_norm = torch.norm(vectors[i] - vectors[j], p=2)
                norm[(i, j)] = l2_norm.detach().cpu().item()
        l2_norms.append(norm)
    return l2_norms

def create_table(l2_norms,basis_num):
    tables = []
    for l in range(len(l2_norms)):
        table = np.zeros((basis_num,basis_num))
        for i in range(basis_num):
            for j in range(i+1,basis_num):
                table[i, j] = l2_norms[l].get((i ,j))
                table[j, i] = l2_norms[l].get((i, j))
        tables.append(table)
    return tables



'''
meta_learner = create_meta_learner(data_seed=242,num_samples=100, epochs=100, intrinsic_dim=400,basis_num=15 ,lr=0.1, seed=137)
print("created meta learner")
layered_params = create_layered_params(meta_learner)
print("created layer params")
basis_num = len(layered_params)
l2_norms = calculate_norm(layered_params)
print("calculated l2 norms")
tables = create_table(l2_norms,basis_num)

filename = 'l2_norms.csv'

with open(filename, 'w') as f:
    for i in range(len(tables)):
        f.write('\n')
        df = pd.DataFrame(tables[i], columns=[f'Vector_{i+1}' for i in range(basis_num)], index=[f'Vector_{i+1}' for i in range(basis_num)])
        f.write(f"--- Layer{i}  ---")
        df.to_csv(f, index=True, header=True) 
        f.write('\n')
          
'''
'''
filename = 'ELLA_Subspace/ELLA_global_params/meta_learner.pt'

loaded_dict = torch.load(filename)
text_dict = {key: value.tolist() for key, value in loaded_dict.items()}

# Save the dictionary as a JSON in a .txt file
output_txt_file = 'ELLA_Subspace/ELLA_global_params/ELLA_global_params.txt'
with open(output_txt_file, 'w') as f:
    json.dump(text_dict, f, indent=4)

print(f"Parameters reloaded and saved to {output_txt_file}")
'''

def entrypoint(**kwargs):
  
  results = main(**kwargs)



if __name__ == '__main__':
  import fire
  fire.Fire(entrypoint)