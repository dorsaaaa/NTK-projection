import os
import copy
import dill
import logging
from datetime import datetime
from random import randint
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm
from pactl.nn.projectors import FixedNumpySeed, FixedPytorchSeed
from pactl.bounds.quantize_fns import quantize_vector
from pactl.bounds.quantize_fns import finetune_quantization, finetune_quantization_all_tasks
from pactl.bounds.quantize_fns import get_message_len
from pactl.bounds.quantize_fns import do_arithmetic_encoding
from pactl.bounds.get_pac_bounds import pac_bayes_bound_opt
from pactl.bounds.get_pac_bounds import compute_catoni_bound
from pactl.train_utils import eval_model, DistributedValue
from pactl.bounds.compute_kl_mixture import get_gains
from pactl.data import get_dataset, get_data_dir
import torch.nn.functional as F
from pathlib import Path
from copy import deepcopy
import json



@torch.no_grad()
def eval_perturbed_model(
    model,
    loader,
    scale=None,
    device=None,
    max_samples=5000,
    distributed=False,
):
    assert scale is not None

    model.eval()

    module = model.module if distributed else model

    orig_weights = copy.deepcopy(module.subspace_params.data)

    acc_samples = []
    std_err = np.inf
    if distributed:
        acc_samples = DistributedValue(acc_samples)

    while std_err > 3e-3:
        for X, Y in tqdm(loader, desc='quantized acc eval'):
            module.subspace_params.data = orig_weights + \
                scale * torch.randn_like(orig_weights)

            X, Y = X.to(device), Y.to(device)

            logits = model(X)

            acc = (logits.argmax(dim=-1) == Y).sum() / len(Y)

            acc_samples += [acc.item()]

        if distributed:
            acc_samples = acc_samples.resolve()

        std_err = np.std(acc_samples) / np.sqrt(len(acc_samples))

        if len(acc_samples) >= max_samples:
            break

        if distributed:
            acc_samples = DistributedValue(acc_samples)

    if distributed:
        acc_samples = acc_samples.resolve()

    module.subspace_params.data = orig_weights
    out = np.mean(acc_samples)

    return out



def compute_quantization_all_tasks(
    meta_learner,
    param_name,
    levels,
    device,
    epochs,
    lr,
    use_kmeans
):
    if levels == 0:
        return None, 0

    learned_centroids = None
    use_finetuning = True if epochs > 0 else False
    vectors = [getattr(model, param_name).cpu().data.numpy() for model in meta_learner.nets]
    length = len(getattr(meta_learner.nets[0], param_name))
    
    if use_finetuning:
        criterion = nn.CrossEntropyLoss()
        qw = finetune_quantization_all_tasks(
            meta_learner=meta_learner,
            param_name=param_name,
            levels=levels,
            device=device,
            epochs=epochs,
            criterion=criterion,
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
        learned_centroids = deepcopy(centroids)
        centroids = centroids.astype(np.float16)
        '''
        #save model
        filename = 'ELLA_Subspace/codebooks.txt'
        dict={}
        dict['centroids']= centroids
        dict['symbols']= symbols
        text_dict = {key: value.tolist() for key, value in dict.items()}
        with open(filename, 'w') as f:
            json.dump(text_dict, f, indent=4)
        print(f"Parameters loaded and saved to {filename}")
        '''
        probabilities = np.array([np.mean(symbols == i) for i in range(levels)])
        _, coded_symbols_size = do_arithmetic_encoding(symbols, probabilities,
                                                       qw.centroids.shape[0])
        message_len = get_message_len(
            coded_symbols_size=coded_symbols_size,
            codebook=centroids,
            max_count=len(symbols),
        )
    else:
        module = model.module if isinstance(model,
                                            torch.nn.parallel.DistributedDataParallel) else model
        vector = sum(vectors, [])
        quantized_vec, message_len = quantize_vector(vector, levels=levels, use_kmeans=use_kmeans)
        
    quantized_vecs = [quantized_vec[length*i:length*(i + 1)] for i in range(len(meta_learner.nets))]
    
    return quantized_vecs, message_len, learned_centroids


def quantize(quant_type = 'default', model = None, param_names = None, levels = None, device = None, train_loader = None, epochs = None, lr = None, use_kmeans = None, partition = None, learned_centroids = None, Transfer=False):  
    quantized_vecs, message_len = None, None

    if quant_type == 'default':
        quantized_vecs, message_len = compute_quantization(model, param_names, levels, device, train_loader, epochs, lr, use_kmeans, partition = partition, learned_centroids = learned_centroids, Transfer=Transfer) 
    #in the meta_learner classes , you cast them finally to float16!
    elif quant_type == 'none' or 'float8' or 'float16' or 'float32' or 'float64':
        message_len = 0
        quantized_vecs = []
        for name in param_names:
            vec, m_len = my_quantize_vector(getattr(model, name).cpu().data.numpy(), quant_type)
            quantized_vecs.append(vec)
            message_len += m_len
    else:
        print("quantize type not implemented!")


    return quantized_vecs, message_len


def compute_quantization(
    model,
    param_names,
    levels,
    device,
    train_loader,
    epochs,
    lr,
    use_kmeans,
    partition = None,
    Transfer=False,
    learned_centroids = None
):
    if levels == 0:
        return None, 0

    use_finetuning = True if epochs > 0 else False
    vectors = [getattr(model, name).cpu().data.numpy() for name in param_names]
    lens = [0] + list(np.cumsum([len(vec) for vec in vectors]))
    
    if use_finetuning:
        criterion = nn.CrossEntropyLoss()
        qw = finetune_quantization(
            model=model,
            param_names=param_names,
            levels=levels,
            device=device,
            train_loader=train_loader,
            epochs=epochs,
            criterion=criterion,
            optimizer='sgd',
            lr=lr,
            use_kmeans=use_kmeans,
            partition = partition,
            Transfer=Transfer,
            learned_centroids = learned_centroids
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
        module = model.module if isinstance(model,
                                            torch.nn.parallel.DistributedDataParallel) else model
        vector = sum(vectors, [])
        quantized_vec, message_len = quantize_vector(vector, levels=levels, use_kmeans=use_kmeans)
        
    quantized_vecs = [quantized_vec[lens[i]:lens[i + 1]] for i in range(len(lens) - 1)]
    
    return quantized_vecs, message_len 


def set_runnings(model,quantized_mean_vec,quantized_var_vec,device):
    slc = 0
    for _, layer in model._forward_net[0].named_modules():
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            length = len(layer.running_mean)
            layer.running_mean = torch.tensor(quantized_mean_vec[slc:slc+length]).float().to(device)
            layer.running_var = torch.tensor(quantized_var_vec[slc:slc+length]).float().to(device)
            slc += length
    return model

def float_to_fp8(num):
    # 1 sign bit, 4 exponent bits, 3 mantissa bits
    if num == 0:
        return 0
    sign = np.sign(num)
    abs_num = np.abs(num)  
    exponent = np.floor(np.log2(abs_num)) 
    mantissa = abs_num / (2**exponent) - 1 
    bias = 7
    fp8_exponent = int(exponent + bias)
    if fp8_exponent < 0:
        return 0.0  
    if fp8_exponent > 15:
        return np.inf * sign 
    fp8_mantissa = int(np.round(mantissa * (2**3))) 
    if fp8_mantissa >= 8:
        fp8_mantissa = 0
        fp8_exponent += 1
    if fp8_exponent > 15:
        return np.inf * sign  # Overflow
    fp8_value = sign * (1 + fp8_mantissa / (2**3)) * (2**(fp8_exponent - bias))
    return fp8_value

def to_float8(vector):
    return np.array([float_to_fp8(x) for x in vector])
    
#DO NOT use this function for default 
def my_quantize_vector(vector,quantize_type):
    typ = None
    bit = None
    if quantize_type == 'none':
        quantize_type = vector.dtype
        

    if quantize_type == 'float16':
        typ = np.float16
        bit = 16
    elif quantize_type == 'float32':
        typ = np.float32
        bit = 32
    elif quantize_type == 'float64':
        typ = np.float64
        bit = 64
    elif quantize_type == 'wrong':
        typ = np.float32
        bit = 0

    quantized_vec = vector.astype(typ)
    
    if quantize_type == 'float8':
        quantized_vec = to_float8(vector)
        bit = 8
    
    message_len = len(vector)*bit
    return quantized_vec, message_len


def evaluate_idmodel(
    model,
    trainloader,
    testloader,
    lr=1.0e-2,
    epochs=10,
    device=torch.device('cuda'),
    posterior_scale=0.01,
    misc_extra_bits=0,
    levels=7,
    use_kmeans=False,
    distributed=False,
    log_dir=None,
    quantize_type = 'default',
    said_quantize_type = None,
    is_said = False
):

    train_acc = eval_model(model, trainloader, device_id=device, distributed=distributed)['acc']
    if log_dir is not None:
        logging.info(f'Train accuracy: {train_acc:.4f}')
    test_acc = eval_model(model, testloader, device_id=device, distributed=distributed)['acc']
    if log_dir is not None:
        logging.info(f'Test accuracy: {test_acc:.4f}')
    is_seperate = True
    if is_said and said_quantize_type == 'default':
        is_seperate = False
    quantized_vec, message_len = compute_quantization(model, levels, device, trainloader, epochs,
                                                      lr, use_kmeans,is_seperate = is_seperate)
    if is_said and said_quantize_type != 'default':
        said_quantized_vec, said_message_len = my_quantize_vector(model.subspace_params[model.d:].cpu().data.numpy(),said_quantize_type) 
        message_len += said_message_len
        quantized_vec = np.append(quantized_vec,said_quantized_vec)
    
    try:
        quantized_mean_vec = None
        quantized_var_vec = None
        module = model.module if distributed else model
        # TODO: Ideally use PyTorch parameters instead of private variable.
        if quantized_vec is not None:
            module.subspace_params.data = torch.tensor(quantized_vec).float().to(device)     
            eval_model(module, trainloader, device_id=0, distributed=False,train_mode = True)
            mean_vec = torch.tensor([]).float().to(device)
            var_vec = torch.tensor([]).float().to(device)
            for name, layer in model._forward_net[0].named_modules():
                if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    mean_vec = torch.cat((mean_vec,layer.running_mean), 0)
                    var_vec = torch.cat((var_vec,layer.running_var), 0)
        
            if quantize_type == 'default':
                quantized_mean_vec, mean_message_len = quantize_vector(mean_vec.cpu().data.numpy(), levels=levels, use_kmeans=use_kmeans) 
                quantized_var_vec, var_message_len = quantize_vector(var_vec.cpu().data.numpy(), levels=levels, use_kmeans=use_kmeans)
            
            else :
                quantized_mean_vec, mean_message_len =my_quantize_vector(mean_vec.cpu().data.numpy(),quantize_type) 
                quantized_var_vec, var_message_len = my_quantize_vector(var_vec.cpu().data.numpy(),quantize_type)
            
            message_len += mean_message_len + var_message_len
            set_runnings(module,quantized_mean_vec,quantized_var_vec,device)
            
        else:
            aux = torch.zeros_like(module.subspace_params.data).float().to(device)
            module.subspace_params.data = aux
    except AttributeError:
        logging.warning("Quantization vector was not updated.")
    
    
    
    
    if log_dir is not None:
        model_params_path = Path(log_dir) / 'quantized_model_params.pt'
        torch.save(module.subspace_params.data.cpu(), model_params_path)
        logging.info(f'Saved quantized model parameters at "{model_params_path}".')

        model_running_mean_path = Path(log_dir) / 'quantized_model_running_mean.pt'
        torch.save(quantized_mean_vec, model_running_mean_path)
        logging.info(f'Saved quantized model running means at "{model_running_mean_path}".')

        model_running_var_path = Path(log_dir) / 'quantized_model_running_var.pt'
        torch.save(quantized_var_vec, model_running_var_path)
        logging.info(f'Saved quantized model running vars at "{model_running_var_path}".')
    #####
    # self delimiting message takes up additional 2 log(l) bits
    prefix_message_len = message_len + 2 * np.log2(message_len) if message_len > 0 else 0
    train_nll_bits = total_nll_bits(model, trainloader, device=device, distributed=distributed)
    # output = eval_acc_and_bound(
    #     model=model, trainloader=trainloader, testloader=testloader,
    #     prefix_message_len=prefix_message_len,
    #     device=device,
    #     quantized_vec=quantized_vec, posterior_scale=posterior_scale)

    raw_output = eval_acc_and_bound(model=model, trainloader=trainloader, testloader=testloader,
                                    prefix_message_len=prefix_message_len, device=device,
                                    misc_extra_bits=misc_extra_bits, quantized_vec=quantized_vec,
                                    posterior_scale=posterior_scale, use_robust_adj=False,
                                    log_dir=log_dir, distributed=distributed)
    raw_output = {f'raw_{name}': value for name, value in raw_output.items()}
    
    

    
    return {
        # **output,
        **raw_output,
        'train_nll_bits': train_nll_bits,
        'prefix_message_len': prefix_message_len,
        'train_err_100': (1. - train_acc) * 100,
        'test_err_100': (1 - test_acc) * 100
    }


@torch.no_grad()
def total_nll_bits(
    model,
    loader,
    device=None,
    distributed=False,
):
    model.eval()

    nll = torch.tensor(0.).to(device)
    if distributed:
        nll = DistributedValue(nll)

    for x, y in tqdm(loader, desc='evaluating nll', leave=False):
        # compute log probabilities and index them by the labels
        logprobs = model(x.to(device)).log_softmax(dim=1)[np.arange(y.shape[0]), y]
        nll += -logprobs.sum().cpu().data.item() / np.log(2)

    if distributed:
        nll = nll.resolve()

    return nll


def eval_acc_and_bound(
    model,
    trainloader,
    testloader,
    prefix_message_len,
    quantized_vec,
    posterior_scale=0.01,
    misc_extra_bits=0.,
    device=None,
    use_robust_adj=True,
    log_dir=None,
    distributed=False,
):
    if log_dir is not None:
        logging.debug('*' * 50 + '\nEvaluating quantized accuracy and getting the bound')

    posterior_scale = None if quantized_vec is None else posterior_scale * np.std(quantized_vec)

    # assert posterior_scale is not None

    if use_robust_adj:
        divergence_gains = get_gains(quantized_vec, posterior_scale)
    else:
        divergence_gains = 0
        # NOTE: Don't pay for posterior scales (we optimize over 4 scales).
        misc_extra_bits -= 2
        posterior_scale = None

    if posterior_scale is None:
        quant_train_acc = eval_model(model, trainloader, device_id=device,
                                     distributed=distributed)['acc']
        if log_dir is not None:
            logging.info(f'Quantized train accuracy: {quant_train_acc:.4f}')

        quant_test_acc = eval_model(model, testloader, device_id=device,
                                    distributed=distributed)['acc']
        if log_dir is not None:
            logging.info(f'Quantized test accuracy: {quant_test_acc:.4f}')
    else:
        ## FIXME: Implement distributed perturbations.
        raise NotImplementedError

    divergence = (prefix_message_len + divergence_gains + misc_extra_bits) * np.log(2)
    train_size = len(trainloader.dataset)
    if quant_train_acc < 0.5:
        err_bound = compute_catoni_bound(train_error=1. - quant_train_acc, divergence=divergence,
                                         sample_size=train_size)
    else:
        err_bound = pac_bayes_bound_opt(divergence=divergence, train_error=1. - quant_train_acc,
                                        n=train_size)
    return {
        'quant_train_err_100': (1 - quant_train_acc) * 100,
        'quant_test_err_100': (1 - quant_test_acc) * 100,
        'divergence_gains': divergence_gains,
        'err_bound_100': err_bound * 100,
        'misc_extra_bits': misc_extra_bits,
    }


def auto_eval(
    model,
    trainloader,
    testloader,
):
    options = [{'use_kmeans': 0, 'levels': 3}, {'use_kmeans': 0, 'levels': 5}]
    best_err = 200
    best_results = None
    best_option = None
    for option in options:
        results = evaluate_idmodel(model, trainloader, testloader, extra_bits=1, epochs=1, **option)
        if results['err_bound_100'] > best_err:
            best_err = results['err_bound_100']
            best_option = option
            best_results = results
    return {**best_results, **best_option}


def get_bounds(
    path,
    dataset,
    data_dir=None,
    subsample=False,
    rescale_posterior=1,
    encoding_type="arithmetic",
    levels=5,
    use_kmeans=0,
    epochs=10,
):
    """path: path to the saved pkl model"""
    rescale_posterior = bool(rescale_posterior)
    logging.getLogger().setLevel(logging.INFO)
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H%M_%S")
    string = 'bound_' + time_stamp + '_' + str(randint(1, int(1.e5)))
    filename = "logs/" + string + ".log"
    logs_exists = os.path.exists('./logs')
    if not logs_exists:
        os.mkdir('./logs')
    logging.basicConfig(filename=filename, level=logging.DEBUG)

    with FixedNumpySeed(0), FixedPytorchSeed(0):
        with open(path, 'rb') as f:
            model = dill.load(f)
    trainset, testset = get_dataset(dataset, root=get_data_dir(data_dir), extra_transform=None,
                                    aug=False)
    trainloader = DataLoader(trainset, batch_size=100, shuffle=False, num_workers=0,
                             pin_memory=False)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0, pin_memory=False)
    use_kmeans = bool(use_kmeans)
    quantize_kwargs = {'use_kmeans': use_kmeans, 'encoding_type': encoding_type, 'levels': levels}
    experiment_code = ''
    if use_kmeans:
        experiment_code += 'k'
    else:
        experiment_code += 'u'
    experiment_code += str(levels)
    experiment_code += str(encoding_type[0])
    results = evaluate_idmodel(model, trainloader, testloader, rescale_posterior=rescale_posterior,
                               subsample=subsample, epochs=epochs, **quantize_kwargs)
    results['code'] = experiment_code
    print(pd.Series(results))
    for key, value in results.items():
        logging.info(f'{key}: {value}')
    results = pd.DataFrame([results])
    results.to_csv(filename[:-4] + '.csv', index=False)

    return results
