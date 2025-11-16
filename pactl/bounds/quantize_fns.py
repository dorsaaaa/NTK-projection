from decimal import Decimal
from copy import deepcopy
import logging
from collections import Counter
from decimal import getcontext
import numpy as np
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
import scipy.stats
import torch
from torch.optim import SGD, Adam
import torch.nn as nn
from pactl.nn.projectors import _delchainattr
from pactl.prune_fns import get_pruned_vec
from itertools import chain


def finetune_prune_quantization(
    model,
    levels,
    device,
    train_loader,
    epochs,
    criterion,
    optimizer,
    lr,
    use_kmeans=False,
):
    vector = get_pruned_vec(model)
    vector = vector.detach().cpu().numpy()
    cluster_fn = get_random_symbols_and_codebook
    if use_kmeans:
        cluster_fn = get_kmeans_symbols_and_codebook
    _, centroids = cluster_fn(vector, levels=levels, codebook_dtype=np.float16)
    has_zero = np.sum(centroids == 0.) > 0
    if not has_zero:
        aux = np.zeros(centroids.shape[0] + 1)
        aux[1:] = centroids
        centroids = aux
    centroids = torch.tensor(centroids, dtype=torch.float32)
    centroids = centroids.to(device)
    print(centroids)
    quantizer_fn = Quantize().apply
    qw = QuantizingWrapperPrune(model, quantizer=quantizer_fn, centroids=centroids)

    optim_params = [qw.centroids]
    if optimizer == "sgd":
        optimizer = SGD(
            optim_params,
            lr=lr,
        )
    elif optimizer == "adam":
        optimizer = Adam(optim_params, lr=lr)
    else:
        raise NotImplementedError

    run_sgd_prunned(train_loader, qw, criterion, optimizer, device=device, epochs=epochs)
    return qw


class QuantizingWrapperPrune(nn.Module):
    def __init__(self, net, quantizer, centroids):
        super().__init__()
        self.subspace_params = []
        self._forward_net = [net]
        self.named_params = list(net.named_parameters())
        for p_name, param in self.named_params:
            aux = nn.Parameter(deepcopy(param), requires_grad=True)
            self.subspace_params.append(aux)
            _delchainattr(net, p_name)
        self.quantizer = quantizer
        self.centroids = nn.Parameter(centroids, requires_grad=True)
        # self.centroids = torch.tensor(centroids, requires_grad=True, dtype=torch.float32)

    def forward(self, *args, **kwargs):
        for idx, (p_name, params) in enumerate(self.named_params):
            aux = self.subspace_params[idx]
            quant_params = self.quantizer(aux.reshape(-1), self.centroids)
            _setchainattr(
                self._forward_net[0], p_name, quant_params.reshape(*aux.shape)
            )
        return self._forward_net[0](*args, **kwargs)



def finetune_quantization(
    model,
    param_names,
    levels,
    device,
    train_loader,
    epochs,
    criterion,
    optimizer,
    lr,
    use_kmeans=False,
    partition = None,
    learned_centroids = None,
    Transfer=False 
):

    vectors = [getattr(model, name).cpu().data.numpy() for name in param_names]
    lens = [0] + list(np.cumsum([len(vec) for vec in vectors]))
    vector = np.array(list(chain.from_iterable(vectors)))
    if not Transfer:
        cluster_fn = get_random_symbols_and_codebook
        if use_kmeans:
            cluster_fn = get_kmeans_symbols_and_codebook
        _, centroids = cluster_fn(vector, levels=levels, codebook_dtype=np.float16)
        centroids = torch.tensor(centroids, dtype=torch.float32).to(device)
    else:
        centroids = torch.tensor(learned_centroids, dtype=torch.float32).to(device)
    
    quantizer_fn = Quantize().apply
    qw = QuantizingWrapper(model, param_names, lens, quantizer=quantizer_fn, centroids=centroids, Transfer=Transfer)
    if optimizer == "sgd":
        params = [qw.subspace_params] if Transfer else [qw.subspace_params, qw.centroids]
        optimizer = SGD(
            params,
            lr=lr,
        )
    elif optimizer == "adam":
        optimizer = Adam([qw.subspace_params, qw.centroids], lr=lr)
    else:
        raise NotImplementedError

        
    
    run_sgd(
        train_loader,
        qw,
        criterion,
        optimizer,
        device=device,
        epochs=epochs,
        partition = partition
    )
    return qw


def finetune_quantization_all_tasks(
    meta_learner,
    param_name,
    levels,
    device,
    epochs,
    criterion,
    optimizer,
    lr,
    use_kmeans=False,
):
    vectors = [getattr(model, param_name).cpu().data.numpy() for model in meta_learner.nets]
    length = len(getattr(meta_learner.nets[0], param_name))

    vector = np.array(list(chain.from_iterable(vectors)))

    cluster_fn = get_random_symbols_and_codebook
    if use_kmeans:
        cluster_fn = get_kmeans_symbols_and_codebook
    _, centroids = cluster_fn(vector, levels=levels, codebook_dtype=np.float16)
    centroids = torch.tensor(centroids, dtype=torch.float32)
    centroids = centroids.to(device)
    quantizer_fn = Quantize().apply
    qw = QuantizingWrapper_all_tasks(meta_learner, param_name, length, quantizer=quantizer_fn, centroids=centroids)
    
    if optimizer == "sgd":
        optimizer = SGD(
            [qw.subspace_params, qw.centroids],
            lr=lr,
        )
    elif optimizer == "adam":
        optimizer = Adam([qw.subspace_params, qw.centroids], lr=lr)
    else:
        raise NotImplementedError


        
    
    run_sgd_all_tasks(
        qw,
        criterion,
        optimizer,
        device=device,
        epochs=epochs,
    )
    return qw



def run_sgd_all_tasks(qw, criterion, optimizer, device=None, epochs=0):
    best_avg_acc_so_far = 0
    qw_subspace_params = deepcopy(qw.subspace_params)
    qw_centroids=deepcopy(qw.centroids)

    for e in tqdm(range(epochs)):
        for i in range(qw.meta_learner.num_train_tasks):
            qw.meta_learner.nets[i].train()

        #print("before: ",qw.subspace_params[:10])
        #print("after: ",qw.subspace_params[:10])


        train_loader_iters = {t: iter(qw.meta_learner.train_loader[t]['train']) for t in range(qw.meta_learner.num_train_tasks)}
        for i in range(qw.meta_learner.num_batches):
            qw.update()
            sum_loss = 0
            optimizer.zero_grad()
            for t in range(qw.meta_learner.num_train_tasks):
                partition = qw.meta_learner.train_loader[t]['partitions'] if not qw.meta_learner.do_mapping else None
                try:
                    (X, Y) = next(train_loader_iters[t])
                    X, Y = X.to(device), Y.to(device)
                    f_hat = qw.meta_learner.nets[t](X)
                    loss = criterion(f_hat[:,partition], Y[:,partition]) if partition is not None else criterion(f_hat, Y)
                    sum_loss += loss
                except StopIteration:
                    train_loader_iters[t] = iter(qw.meta_learner.train_loader[t]['train'])
            #print("sum_loss:", sum_loss)
            sum_loss.backward()
            optimizer.step()

        #eval
        acc=0
        for t in range(qw.meta_learner.num_train_tasks):
            partition = qw.meta_learner.train_loader[t]['partitions'] if not qw.meta_learner.do_mapping else None
            train_acc = evaluate(qw.meta_learner.nets[t], qw.meta_learner.train_loader[t]['train'], device_id=device, partition = partition)
            acc += train_acc


        if acc > best_avg_acc_so_far:
            best_avg_acc_so_far = acc
            qw_subspace_params = deepcopy(qw.subspace_params)
            qw_centroids = deepcopy(qw.centroids)
            #print("best acc: ",best_acc_so_far)
            #print("sp:", qw_subspace_params)
            #print("centroids:", qw_centroids)


    qw.subspace_params = qw_subspace_params
    qw.centroids = qw_centroids
            
          

def run_sgd_prunned(
    train_loader,
    net,
    criterion,
    optim,
    device=None,
    epochs=0,
):

    for e in tqdm(range(epochs)):
        net.train()
        logging.debug(f"centroids: {net.centroids}")
        for i, (X, Y) in tqdm(enumerate(train_loader), leave=False):
            X, Y = X.to(device), Y.to(device)
            optim.zero_grad()
            f_hat = net(X)
            loss = criterion(f_hat, Y)
            loss.backward()
            net.centroids.grad[0] = 0.
            optim.step()

            if i % 100 == 0:
                metrics = {"epoch": e, "mini_loss": loss.detach().item()}
                logging.info(metrics, extra=dict(wandb=True, prefix="sgd/train"))



def run_sgd(
    train_loader,
    net,
    criterion,
    optim,
    device=None,
    epochs=0,
    partition = None
):
    best_acc_so_far = 0
    qw_subspace_params = deepcopy(net.subspace_params)
    qw_centroids= deepcopy(net.centroids)

    
    for e in tqdm(range(epochs)):
        net.train()
        
        for i, (X, Y) in tqdm(enumerate(train_loader), leave=False):
            
            X, Y = X.to(device), Y.to(device)

            optim.zero_grad()
            f_hat = net(X)
            
            loss = criterion(f_hat[:,partition], Y[:,partition]) if partition is not None else criterion(f_hat, Y)

            loss.backward()
            optim.step()
            
        #eval
        acc = evaluate(net, train_loader,device, partition = partition)

        if acc > best_acc_so_far:
            best_acc_so_far = acc
            qw_subspace_params = deepcopy(net.subspace_params)
            qw_centroids = deepcopy(net.centroids)
            #print("best acc: ",best_acc_so_far)
            #print("sp:", qw_subspace_params)
            #print("centroids:", qw_centroids)


    net.subspace_params = qw_subspace_params
    net.centroids = qw_centroids
            
        

   


@torch.no_grad()
def evaluate(model, data_loader, device_id=None, partition=None):
    model.eval()
    N = len(data_loader.dataset)
    N_acc = 0
    

    for i, (X, Y) in enumerate(data_loader):
        
        X, Y = X.to(device_id), Y.to(device_id)       
        logits = model(X) 
        if partition is not None :
            mask = logits[:, partition]
            max_indices_partition = mask.argmax(dim=-1)
            max_indices = torch.tensor(partition, device=device_id)[max_indices_partition]
        else:
          max_indices = logits.argmax(dim=-1)
        N_acc += (max_indices == Y.argmax(dim=-1)).sum() if Y.ndim > 1 else (max_indices == Y).sum()
    
    acc= N_acc.item() / N
    return acc

def create_assigment_matrix_prune(labels, num_clusters):
    assignments = torch.zeros(size=(num_clusters,) + labels.shape, device=labels.device)
    for k in range(num_clusters):
        assignments[k, labels == k] = 1.0
    return assignments


class QuantizingWrapper_all_tasks(nn.Module):
    def __init__(self, meta_learner, param_name, length, quantizer, centroids):
        super().__init__()
        vectors = [getattr(net, param_name) for net in meta_learner.nets]
        self.subspace_params = deepcopy(
            nn.Parameter(torch.cat(vectors, dim=0), requires_grad=True)
        )
        for net in meta_learner.nets:
            _delchainattr(net, param_name)
        self.param_name = param_name
        self.length = length
        self.meta_learner = meta_learner
        self.quantizer = quantizer
        self.centroids = nn.Parameter(centroids, requires_grad=True)

    
    def to(self, *args, **kwargs):
        #self._forward_net[0].to(*args, **kwargs)
        return super().to(*args, **kwargs)
    

    def update(self, *args, **kwargs):
        for i in range(len(self.meta_learner.nets)):
            _setchainattr(
                self.meta_learner.nets[i],
                self.param_name,
                self.quantizer(self.subspace_params[self.length * i:self.length*(i+1)], self.centroids),
            )
        
    

class QuantizingWrapper(nn.Module):
    def __init__(self, net, param_names, lens, quantizer, centroids, Transfer=False):
        super().__init__()
        vectors = [getattr(net, name) for name in param_names]
        self.subspace_params = deepcopy(
            nn.Parameter(torch.cat(vectors, dim=0), requires_grad=True)
        )
        for name in param_names:
            _delchainattr(net, name)
        self.param_names = param_names
        self.lens = lens
        self._forward_net = [net]
        self.quantizer = quantizer
        self.centroids = nn.Parameter(centroids, requires_grad=not Transfer) 

    def to(self, *args, **kwargs):
        self._forward_net[0].to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, *args, **kwargs):
        for i in range(len(self.param_names)):
            _setchainattr(
                self._forward_net[0],
                self.param_names[i],
                self.quantizer(self.subspace_params[self.lens[i]:self.lens[i+1]], self.centroids),
            )
        
        if self.training:
            self._forward_net[0].train()
        else:
            self._forward_net[0].eval()
        return self._forward_net[0](*args, **kwargs)



class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, centroids):
        vec = (centroids.unsqueeze(-2) - params.unsqueeze(-1)) ** 2.0
        mask = torch.min(vec, -1)[-1]
        ctx.assignment = create_assigment_matrix(mask, centroids.shape[0])
        quantized_params = centroids[mask]
        return quantized_params

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, ctx.assignment @ grad_output


def create_assigment_matrix(labels, num_clusters):
    size = (num_clusters, labels.shape[0])
    assignments = torch.zeros(size=size, device=labels.device)
    for k in range(num_clusters):
        assignments[k, labels == k] = 1.0
    return assignments


def quantize_vector(
    vec, levels=2**2 + 1, use_kmeans=False, encoding_type="arithmetic"
):
    codebook_dtype = np.float16
    if use_kmeans:
        symbols, codebook = get_kmeans_symbols_and_codebook(vec, levels, codebook_dtype)
    else:
        symbols, codebook = get_random_symbols_and_codebook(vec, levels, codebook_dtype)

    #logging.info(f"KMeans: {use_kmeans}, Levels: {levels}, Algorithm: {encoding_type}")
    probabilities = np.array([np.mean(symbols == i) for i in range(levels)])
    #logging.info(f"probs {probabilities}")

    if encoding_type == "arithmetic":
        _, coded_symbols_size = do_arithmetic_encoding(
            symbols, probabilities, levels
        )
    elif encoding_type == "huff":
        _, coded_symbols_size = do_huffman_encoding(symbols)
    else:
        NotImplementedError
    decoded_vec = np.zeros(shape=(len(vec)))
    for k in range(len(codebook)):
        decoded_vec[symbols == k] = codebook[k]

    message_len = get_message_len(coded_symbols_size, codebook, len(symbols))
    #logging.info(f"Message Len: {message_len}")
    return decoded_vec, message_len


def get_random_symbols_and_codebook(vec, levels, codebook_dtype):
    largest = max(np.max(vec), np.abs(np.min(vec)))
    initvals = np.linspace(-largest - 1e-6, largest + 1e-6, levels + 1)
    assignments = np.digitize(vec, initvals) - 1
    centroids = []
    for i in range(levels):
        aux = vec[assignments == i]
        if len(aux) > 0:
            centroids.append(np.mean(aux))
        else:
            centroids.append(initvals[i])
    codebook = np.array(centroids, dtype=codebook_dtype)
    symbols = np.array(assignments)
    return symbols, codebook


def get_kmeans_symbols_and_codebook(vec, levels, codebook_dtype):
    kmeans = KMeans(n_clusters=levels).fit(vec.reshape(-1, 1))
    codebook = kmeans.cluster_centers_.astype(codebook_dtype)[:, 0]
    symbols = kmeans.labels_
    return symbols, codebook


def get_message_len(coded_symbols_size, codebook, max_count):
    codebook_bits_size = 16 if codebook.dtype == np.float16 else 32
    #print("bit size: ",codebook_bits_size)
    probability_bits = int(np.ceil(np.log2(max_count)) * len(codebook))
    codebook_bits = len(codebook) * codebook_bits_size
    summary = f"encoding {coded_symbols_size}, codebook {codebook_bits} probs {probability_bits}"
    #logging.info(summary)
    message_len = coded_symbols_size + codebook_bits + probability_bits
    return message_len


def do_huffman_encoding(vec):
    vec_str = ""
    for i in range(len(vec)):
        vec_str += str(vec[i])
    freq = dict(Counter(vec_str))
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    node = make_tree(freq)
    encoding = huffman_code_tree(node)

    coded_symbols_len = 0
    for i in range(len(vec)):
        key = str(vec[i])
        key_size = len(encoding[key])
        coded_symbols_len += key_size
    return encoding, coded_symbols_len


def do_arithmetic_encoding(symbols, probabilities, levels):
    entropy_est = scipy.stats.entropy(probabilities, base=2)
    #logging.info(f"Entropy: {entropy_est:.2f} bits")
    is_too_large_to_run = len(symbols) > int(1e4)
    if is_too_large_to_run:
        coded_symbols_size = np.ceil(len(symbols) * entropy_est) + 1.
    else:
        getcontext().prec = int(1.1 * np.log10(levels) * len(symbols))
        coded_symbols_size = len(encode(symbols, probabilities))
    return symbols, coded_symbols_size


def decimal2bits(decimal, bits_encoded):
    output_bits = []
    while len(output_bits) < bits_encoded:
        if decimal > Decimal(1) / Decimal(2):
            output_bits.append(1)
            decimal -= Decimal(1) / Decimal(2)
        else:
            output_bits.append(0)
        decimal *= Decimal(2)
    return output_bits


def bits2decimal(bits):
    val = Decimal(0)
    for i, bit in enumerate(bits):
        val += bit * Decimal(2) ** (-(i + 1))
    return val


def encode(sequence, probs):
    """Arithmetic coding of sequence of integers Seq: [a0,a1,a2,...]
    with probabilities: [c0,c1,c2,...]"""
    cumulative_probs = np.cumsum(probs)
    width = Decimal(1)
    message_value = Decimal(0)
    bits_encoded = 0
    for i, val in enumerate(sequence):
        bin_start = cumulative_probs[val - 1] if val > 0 else 0.0
        bin_size = probs[val]
        message_value = message_value + Decimal(bin_start) * width
        width = width * Decimal(bin_size)
        bits_encoded -= np.log2(bin_size)
    #logging.info(f"arithmetic encoded bits {bits_encoded:.2f}")
    return decimal2bits(message_value + width / 2, np.ceil(bits_encoded) + 1)


def decode(bits, probs, N):
    """Arithmetic decoder which decodes bitstream using probabilities: [c0,c1,c2,...]"""
    message_val = bits2decimal(bits)
    cumulative_probs = np.cumsum(probs)
    width = Decimal(1)
    decoded_vals = []
    for i in range(N):
        bin_id = np.digitize(float(message_val), cumulative_probs)
        bin_start = cumulative_probs[bin_id - 1] if bin_id > 0 else 0.0
        bin_size = probs[bin_id]

        message_val = (message_val - Decimal(bin_start)) / Decimal(bin_size)
        width = width * Decimal(bin_size)
        decoded_vals.append(bin_id)
    return decoded_vals


class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right

    def __str__(self):
        return self.left, self.right


def _setchainattr(obj, attr, value):
    attributes = attr.split(".")
    for a in attributes[:-1]:
        obj = getattr(obj, a)
    # FIXME: not everything has to be a param
    # setattr(obj, attributes[-1], nn.Parameter(value))
    setattr(obj, attributes[-1], value)


def huffman_code_tree(node, binString=""):
    """
    Function to find Huffman Code
    """
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, binString + "0"))
    d.update(huffman_code_tree(r, binString + "1"))
    return d


def make_tree(nodes):
    """
    Function to make tree
    :param nodes: Nodes
    :return: Root of the tree
    """
    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    return nodes[0][0]


if __name__ == "__main__":
    vec = np.array([1, 2, 0, 0, 3, 3, 3, 2, 2, 0, 2, 0, 2, 0, 2])
    encoding = do_huffman_encoding(vec)
    for i in encoding:
        print(f"{i} : {encoding[i]}")
