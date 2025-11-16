import sys
import os 
sys.path.append(os.getcwd())
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from data.Utils.data_gen import Task_Generator, get_next_batch_cyclic
import numpy as np
import random, math, pickle
from data.Data_Path import get_data_path
from torch.optim import SGD, Adam
from data.Utils.common import set_random_seed


def get_next_batch_cyclic(data_iterator, task_id, data_generator):
    ''' get sample from iterator, if it finishes then restart  '''
    try:
        batch_data = next(data_iterator[task_id])
    except StopIteration:
        # in case some task has less samples - just restart the iterator and re-use the samples
        data_iterator[task_id] = iter(data_generator)
        batch_data = next(data_iterator[task_id])
    return batch_data




prm = type('', (), {})()
prm.run_name = 'Shuffled_' #@param {type:"string"}
prm.gpu_index = 0 #@param {type:"integer"}
prm.seed = 1 #@param {type:"integer"}
prm.mode = 'MetaTrain' #@param ["MetaTrain", "LoadMetaModel"] {type:"string"}
prm.load_model_path = '' #@param {type:"string"}
prm.test_batch_size = 512 #@param {type:"integer"}
prm.n_test_tasks = 20 #@param {type:"integer"}
prm.data_source = 'MNIST' #@param ["MNIST", "CIFAR10", "Omniglot", "SmallImageNet"] {type:"string"}
prm.n_train_tasks = 10 #@param {type:"integer"}
prm.data_transform = 'Shuffled_Pixels' #@param ["None", "Permute_Pixels", "Permute_Labels", "Shuffled_Pixels"] {type:"string"}
prm.n_pixels_shuffles = 200 #@param {type:"integer"}
prm.limit_train_samples_in_train_tasks = 600 #@param {type:"integer"}
prm.limit_train_samples_in_test_tasks = 100 #@param {type:"integer"}
prm.N_Way = 5 #@param {type:"integer"}
prm.K_Shot_MetaTrain = 100 #@param {type:"integer"}
prm.K_Shot_MetaTest = 100 #@param {type:"integer"}
prm.n_meta_train_classes = 500 #@param {type:"integer"}
prm.chars_split_type = 'random' #@param ["random", "predefined_split"] {type:"string"}
prm.n_meta_train_chars = 1200 #@param {type:"integer"}
prm.complexity_type = 'Seeger' #@param ["NoComplexity", "Variational_Bayes", "PAC_Bayes_Pentina", "McAllester", "Seeger"] {type:"string"}
prm.loss_type = 'CrossEntropy' #@param ["CrossEntropy", "L2_SVM"] {type:"string"}
prm.model_name = 'FcNet3' #@param ["OmConvNet", "FcNet3", "ConvNet3"] {type:"string"}
prm.batch_size = 128 #@param {type:"integer"}
prm.n_meta_train_epochs = 100 #@param {type:"integer"}
prm.n_inner_steps = 50 #@param {type:"integer"}
prm.n_meta_test_epochs = 100 #@param {type:"integer"}
prm.lr = 0.001 #@param {type:"number"}
prm.meta_batch_size = 5 #@param {type:"integer"}
prm.init_from_prior = True #@param {type:"boolean"}
prm.log_var_init = {'mean': -10, 'std': 0.1}
prm.n_MC = 1 #@param {type:"integer"}
prm.optim_func = 'Adam' #@param ["Adam", "SGD"] {type:"string"}
prm.optim_args = {'lr': 0.001}
prm.lr_schedule = {}
prm.kappa_prior = 100.0 #@param {type:"number"}
prm.kappa_post = 0.001 #@param {type:"number"}
prm.delta = 0.1 #@param {type:"number"}
prm.test_type = 'MaxPosterior' #@param ["MaxPosterior", "MajorityVote", "AvgVote"] {type:"string"}
prm.data_path = get_data_path()



def data_gen(dataset='MNIST', task_name='Shuffled', n_train_tasks=10, n_test_tasks=20, samples_per_train=600, samples_per_test=600, seed = 42, data_transform = 'Shuffled_Pixels_Permute_Labels',
        num_classes=40, K_Shot_MetaTrain=15, K_Shot_MetaTest=15):
    set_random_seed(seed)
    #print(dataset, task_name)
    if dataset == 'MNIST' and task_name == 'Shuffled':
        prm.data_transform = data_transform
        prm.limit_train_samples_in_train_tasks = samples_per_train
        prm.limit_train_samples_in_test_tasks = samples_per_test
        prm.n_train_tasks = n_train_tasks
        limit_train_samples_in_train_tasks = prm.limit_train_samples_in_train_tasks
        prm.n_test_tasks = n_test_tasks
        limit_train_samples_in_test_tasks = prm.limit_train_samples_in_test_tasks

        task_generator = Task_Generator(prm)
        train_data_loaders = task_generator.create_meta_batch(prm, n_train_tasks, limit_train_samples=limit_train_samples_in_train_tasks, meta_split='meta_train')
        test_tasks_data = task_generator.create_meta_batch(prm, n_test_tasks, limit_train_samples=limit_train_samples_in_test_tasks, meta_split='meta_test')
        
        n_train_tasks = len(train_data_loaders)
        train_iterators = [iter(train_data_loaders[i]['train']) for i in range(n_train_tasks)]
        
        
        return train_data_loaders, test_tasks_data

    elif dataset=='Omniglot':
        prm.batch_size = 64
        prm.data_source= 'Omniglot'
        prm.data_transform = 'None'
        #prm.chars_split_type
        #prm.n_meta_train_chars
        prm.N_Way = num_classes #number of labels (chars) in the task.
        prm.K_Shot_MetaTrain = K_Shot_MetaTrain #sample this many training examples from each char class,rest of the char examples will be in the test set.
        prm.K_Shot_MetaTest = K_Shot_MetaTest 
        prm.limit_train_samples_in_train_tasks = samples_per_train
        prm.limit_train_samples_in_test_tasks = samples_per_test
        prm.n_train_tasks = n_train_tasks  
        prm.n_test_tasks = n_test_tasks
        
        task_generator = Task_Generator(prm)
        train_data_loaders = task_generator.create_meta_batch(prm, n_train_tasks, limit_train_samples=prm.limit_train_samples_in_train_tasks, meta_split='meta_train')
        test_tasks_data = task_generator.create_meta_batch(prm, n_test_tasks, limit_train_samples=prm.limit_train_samples_in_test_tasks, meta_split='meta_test')

        n_train_tasks = len(train_data_loaders)
        train_iterators = [iter(train_data_loaders[i]['train']) for i in range(n_train_tasks)]


        return train_data_loaders, test_tasks_data
    
    elif dataset=='CIFAR10':
        prm.data_source= 'CIFAR10'
        prm.data_transform = 'None'
        
        
        prm.limit_train_samples_in_train_tasks = None#samples_per_train
        prm.limit_train_samples_in_test_tasks = None#samples_per_test
        prm.n_train_tasks = n_train_tasks  
        prm.n_test_tasks = n_test_tasks
        
        task_generator = Task_Generator(prm)
        train_data_loaders = task_generator.create_meta_batch(prm, n_train_tasks, limit_train_samples=prm.limit_train_samples_in_train_tasks, meta_split='meta_train')
        test_tasks_data = task_generator.create_meta_batch(prm, n_test_tasks, limit_train_samples=prm.limit_train_samples_in_test_tasks, meta_split='meta_test')

        n_train_tasks = len(train_data_loaders)
        train_iterators = [iter(train_data_loaders[i]['train']) for i in range(n_train_tasks)]


        return train_data_loaders, test_tasks_data
      

