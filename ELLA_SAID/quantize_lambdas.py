import os
import copy
import dill
import logging
from datetime import datetime
from random import randint
import numpy as np
import pandas as pd
from copy import deepcopy
from torch.optim import SGD, Adam
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm
from pactl.nn.projectors import FixedNumpySeed, FixedPytorchSeed
from pactl.bounds.quantize_fns import quantize_vector
from pactl.bounds.quantize_fns import get_message_len
from pactl.bounds.quantize_fns import do_arithmetic_encoding
from pactl.bounds.get_pac_bounds import pac_bayes_bound_opt
from pactl.bounds.get_pac_bounds import compute_catoni_bound
from pactl.train_utils import eval_model, DistributedValue
from pactl.bounds.compute_kl_mixture import get_gains
from pactl.data import get_dataset, get_data_dir
import torch.nn.functional as F
from pathlib import Path
from pactl.bounds.quantize_fns import get_random_symbols_and_codebook, get_kmeans_symbols_and_codebook, Quantize, _setchainattr, evaluate
from pactl.nn.projectors import _delchainattr
from pactl.bounds.get_bound_from_chk_v2 import my_quantize_vector
from itertools import chain


def quantize_lambdas(quant_type = 'default', meta_learner = None, levels = None, device = None, epochs = None, lr = None, use_kmeans = None):  
    quantized_vec, message_len = None, None

    if quant_type == 'default':
        quantized_vec, message_len = compute_meta_quantization(meta_learner, levels, device, epochs, lr, use_kmeans) 
    elif quant_type == 'none' or 'float8' or 'float16' or 'float32' or 'float64':
        quantized_vec, message_len = my_quantize_vector(meta_learner.global_param.cpu().data.numpy(), quant_type)
    else:
        print("quantize type not implemented!")

    if quantized_vec is None:
        print("error: global param is none!")
    return quantized_vec, message_len


def compute_meta_quantization(
    meta_learner,
    levels,
    device,
    epochs,
    lr,
    use_kmeans,
):
    if levels == 0:
        return None, 0
    
    vectors = [meta_learner.global_lambdas.cpu().data.numpy(), meta_learner.task_lambdas.cpu().data.numpy()]
    
    use_finetuning = True if epochs > 0 else False
    if use_finetuning:
        ## FIXME: for distributed training.
        criterion = nn.CrossEntropyLoss()
        qw = meta_finetune_quantization(
            meta_learner=meta_learner,
            levels=levels,
            device=device,
            epochs=epochs,
            criterion=criterion,
            # optimizer='adam',
            optimizer='sgd',
            lr=lr,
            use_kmeans=use_kmeans,
        )
        quantized_vec = qw.quantizer(qw.subspace_params, qw.centroids)
        quantized_vec = quantized_vec.cpu().detach().numpy()

        vec = (qw.centroids.unsqueeze(-2) - qw.subspace_params.unsqueeze(-1))**2.0
        symbols = torch.min(vec, -1)[-1]
        symbols = symbols.cpu().detach().numpy()

        centroids = qw.centroids.cpu().detach().numpy()
        centroids = centroids.astype(np.float16)
        probabilities = np.array([np.mean(symbols == i) for i in range(levels)])
        _, coded_symbols_size = do_arithmetic_encoding(symbols, probabilities,
                                                       qw.centroids.shape[0])
        message_len = get_message_len(
            coded_symbols_size=coded_symbols_size,
            codebook=centroids,
            max_count=len(symbols),
        )
    else:
        vector = sum(vectors, [])
        quantized_vec, message_len = quantize_vector(vector, levels=levels, use_kmeans=use_kmeans)
        
    quantized_vecs = [quantized_vec[:meta_learner.said_size],quantized_vec[meta_learner.said_size:]]
    
    return quantized_vecs, message_len

def meta_finetune_quantization(
    meta_learner,
    levels,
    device,
    epochs,
    criterion,
    optimizer,
    lr,
    use_kmeans=True,
):

    vectors = [meta_learner.global_lambdas.cpu().data.numpy(), meta_learner.task_lambdas.cpu().data.numpy()]
    vector = np.array(list(chain.from_iterable(vectors)))

    cluster_fn = get_random_symbols_and_codebook
    if use_kmeans:
        cluster_fn = get_kmeans_symbols_and_codebook
    _, centroids = cluster_fn(vector, levels=levels, codebook_dtype=np.float16)
    centroids = torch.tensor(centroids, dtype=torch.float32)
    centroids = centroids.to(device)
    quantizer_fn = Quantize().apply
    qw = MetaQuantizingWrapper(meta_learner, quantizer=quantizer_fn, centroids=centroids)
    
    if optimizer == "sgd":
        params = [qw.subspace_params, qw.centroids]

        optimizer = SGD(
            params,
            lr=lr,
        )
    elif optimizer == "adam":
        optimizer = Adam([qw.subspace_params, qw.centroids], lr=lr)
    else:
        raise NotImplementedError

    
        
    
    run_sgd(
        qw,
        criterion,
        optimizer,
        device=device,
        epochs=epochs
    )

    return qw

def run_sgd(
    qw,
    criterion,
    optimizer,
    device=None,
    epochs=0
):  
    meta_learner = qw._forward_net[0]
    best_avg_acc_so_far = 0
    qw_subspace_params = None
    qw_centroids=None


    for e in tqdm(range(epochs)):


        train_loader_iters = [iter(meta_learner.train_loader[t]['train']) for t in range(meta_learner.num_train_tasks)]
        for i in range(meta_learner.num_batches):
            qw.update_lambdas()
            sum_loss = 0
            optimizer.zero_grad()
            for t in range(meta_learner.num_train_tasks):
                (X, Y) = next(train_loader_iters[t])
                X, Y = X.to(device), Y.to(device)
                f_hat = meta_learner.nets[t](X)
                loss = criterion(f_hat, Y)
                sum_loss += loss
            sum_loss.backward()
            optimizer.step()
        

        #eval
        acc=0
        for t in range(meta_learner.num_train_tasks):
            train_acc = evaluate(meta_learner.nets[t], meta_learner.train_loader[t]['train'], device_id=device)
            acc += train_acc
        acc = acc/meta_learner.num_train_tasks

        if acc > best_avg_acc_so_far:
            best_avg_acc_so_far = acc
            qw_subspace_params = deepcopy(qw.subspace_params)
            qw_centroids = deepcopy(qw.centroids)
            #print("best acc: ",best_acc_so_far)
            #print("sp:", qw_subspace_params)
            #print("centroids:", qw_centroids)


    qw.subspace_params = qw_subspace_params
    qw.centroids = qw_centroids
            
        
    
     
class MetaQuantizingWrapper(nn.Module):
    def __init__(self, meta_learner, quantizer, centroids):
        super().__init__()

        vectors = [meta_learner.global_lambdas, meta_learner.task_lambdas]
        self.subspace_params = deepcopy(
            nn.Parameter(torch.cat(vectors, dim=0), requires_grad=True)
        )
        
        _delchainattr(meta_learner, "global_lambdas")
        _delchainattr(meta_learner, "task_lambdas")
        self._forward_net = [meta_learner]
        self.quantizer = quantizer
        self.centroids = nn.Parameter(centroids, requires_grad=True)

    def update_lambdas(self, *args, **kwargs):
        _setchainattr(
        self._forward_net[0],
        "global_lambdas",
        self.quantizer(self.subspace_params[:self._forward_net[0].said_size], self.centroids),)
        
        _setchainattr(
        self._forward_net[0],
        "task_lambdas",
        self.quantizer(self.subspace_params[self._forward_net[0].said_size:], self.centroids),)
        
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
   