import torch
import numpy as np
import math
import torch.nn as nn
from copy import deepcopy
import random
from functools import partial
from .linear_operator_base import (
    Lazy,
    LazyKron,
    ConcatLazy,
    LazyPerm,
    LinearOperator,
    LazyDirectSum,
)


_DEFAULT_SEED = 137


def _getchainattr(obj, attr):
    attributes = attr.split(".")
    for a in attributes:
        obj = getattr(obj, a)
    return obj


def _delchainattr(obj, attr):
    attributes = attr.split(".")
    for a in attributes[:-1]:
        obj = getattr(obj, a)
    try:
        delattr(obj, attributes[-1])
    except AttributeError:
        raise


def _setchainattr(obj, attr, value):
    attributes = attr.split(".")
    for a in attributes[:-1]:
        obj = getattr(obj, a)
    setattr(obj, attributes[-1], value)


def flatten(tensorList):
    flatList = []
    for t in tensorList:
        flatList.append(t.contiguous().view(t.numel()))
    return torch.cat(flatList)


def unflatten_like(vector, likeTensorList):
    outList = []
    i = 0
    for tensor in likeTensorList:
        n = tensor.numel()
        outList.append(vector[i: i + n].view(tensor.shape))
        i += n
    return outList


class QuantizingWrapper(nn.Module):
    def __init__(self, net, centroids, assignments):
        super().__init__()
        self.subspace_params = deepcopy(net.subspace_params)
        _delchainattr(net, "subspace_params")

        self._forward_net = [net]
        self.centroids = [centroids]
        self.assignments = assignments

    def to(self, *args, **kwargs):
        self._forward_net[0].to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, *args, **kwargs):
        _setchainattr(self._forward_net[0], "subspace_params", self.subspace_params)
        return self._forward_net[0](*args, **kwargs)


class FixedPytorchSeed(object):
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.pt_rng_state = torch.random.get_rng_state()
        self.cuda_rng_state = torch.cuda.get_rng_state_all()
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    def __exit__(self, *_):
        torch.random.set_rng_state(self.pt_rng_state)
        torch.cuda.set_rng_state_all(self.cuda_rng_state)


class FixedNumpySeed(object):
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.np_rng_state = np.random.get_state()
        np.random.seed(self.seed)
        self.rand_rng_state = random.getstate()
        random.seed(self.seed)

    def __exit__(self, *args):
        np.random.set_state(self.np_rng_state)
        random.setstate(self.rand_rng_state)


class IDModule(nn.Module):

    def __init__(self, net, projector, dimension=1000, global_param = None, global_P = None, groups = None, is_lambda_param = True, is_last_layer_only=False):
        super().__init__()
        self.d = dimension
        self.global_param = global_param
        self._forward_net = [net]
        initnet = deepcopy(net)
        for orig_name, orig_p in initnet.named_parameters():
            if orig_p.requires_grad:
                _delchainattr(net, orig_name)
                
        aux = [(n, p) for n, p in initnet.named_parameters() if p.requires_grad]
        
        self.names, self.trainable_initparams = zip(*aux)
        self.trainable_initparams = [param for param in self.trainable_initparams]
        self.names = list(self.names)
        
        self.D = sum([param.numel() for param in self.trainable_initparams])
        self.subspace_params = nn.Parameter(torch.zeros(self.d))
        self.P = projector(self.D, self.d, self.trainable_initparams, self.names)
        


    def to(self, *args, **kwargs):
        self._forward_net[0].to(*args, **kwargs)
        self.trainable_initparams = [
            param.to(*args, **kwargs) for param in self.trainable_initparams
        ]
        return super().to(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.training:
            self._forward_net[0].train()
        else:
            self._forward_net[0].eval()

        if self.global_param == None:
            self.global_param = torch.zeros(self.d, device = self.subspace_params.get_device())  

        flat_projected_params = self.P @ (self.subspace_params + self.global_param)


        unflattened_params = unflatten_like(
            flat_projected_params, self.trainable_initparams
        )
        iterables = zip(self.names, self.trainable_initparams, unflattened_params)
       
        for p_name, init, proj_param in iterables:
            p = init + proj_param.view(*init.shape)
            _setchainattr(self._forward_net[0], p_name, p)
        return self._forward_net[0](*args, **kwargs)

class SAIDModule(nn.Module):
    
    def __init__(self, net, projector, dimension=1000, global_param = None, global_P = None, groups = None, is_lambda_param = True, is_last_layer_only=False):
        super().__init__()
        self._forward_net = [net]
        self.global_param = global_param
        initnet = deepcopy(net)
        for orig_name, orig_p in initnet.named_parameters():
            if orig_p.requires_grad:
                _delchainattr(net, orig_name)
                
        self.said_size = len([p for n,p in initnet.named_parameters() if p.requires_grad])
        self.d = dimension - self.said_size    
     
        aux = [(n, p) for n, p in initnet.named_parameters() if p.requires_grad]
        
        self.names, self.trainable_initparams = zip(*aux)
        self.trainable_initparams = [param for param in self.trainable_initparams]
        self.names = list(self.names)
        
        self.D = sum([param.numel() for param in self.trainable_initparams])
        self.subspace_params = nn.Parameter(torch.zeros(self.d))
        self.lambdas = nn.Parameter(torch.ones(self.said_size))
        self.P = projector(self.D, self.d, self.trainable_initparams, self.names)
        

    def to(self, *args, **kwargs):
        self._forward_net[0].to(*args, **kwargs)
        self.trainable_initparams = [
            param.to(*args, **kwargs) for param in self.trainable_initparams
        ]
        return super().to(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.training:
            self._forward_net[0].train()
        else:
            self._forward_net[0].eval()


        if self.global_param == None:
            self.global_param = torch.zeros(self.d, device = self.subspace_params.get_device())

        flat_projected_params = self.P @ (self.subspace_params + self.global_param)

        unflattened_params = unflatten_like(
            flat_projected_params, self.trainable_initparams
        )
        iterables = zip(range(self.said_size),self.names, self.trainable_initparams, unflattened_params)

        for i,p_name, init, proj_param in iterables:
            p = init + (self.lambdas[i] * proj_param.view(*init.shape))
            _setchainattr(self._forward_net[0], p_name, p)
        return self._forward_net[0](*args, **kwargs)

def count_el(list):
    size = 0
    for l in list:
        size += l.numel()
    return size

class MetaSAIDModule(nn.Module):
    
    def __init__(self, net, projector, dimension=1000, global_param = None, global_P = None, groups = None, is_lambda_param = True, is_last_layer_only=False):
        super().__init__()
        self._forward_net = [net]
        self.global_param = global_param
        self.global_P = global_P
        self.is_lambda_param = is_lambda_param
        initnet = deepcopy(net)
        for orig_name, orig_p in initnet.named_parameters():
            if orig_p.requires_grad:
                _delchainattr(net, orig_name)
                
        self.said_size = len([p for n,p in initnet.named_parameters() if p.requires_grad])
        self.d = dimension - self.said_size  

        last_layer_lambda = torch.ones(self.said_size)
        last_layer_lambda[self.said_size-2:] = torch.zeros(2)
        self.global_lambdas = nn.Parameter(torch.ones(self.said_size)) if is_lambda_param else (last_layer_lambda if is_last_layer_only else torch.ones(self.said_size))
     
        aux = [(n, p) for n, p in initnet.named_parameters() if p.requires_grad]
        
        self.names, self.trainable_initparams = zip(*aux)
        self.trainable_initparams = [param for param in self.trainable_initparams]
        self.names = list(self.names)
        
        self.D = sum([param.numel() for param in self.trainable_initparams])
        self.subspace_params = nn.Parameter(torch.ones(self.d)/math.sqrt(self.d))

        last_layer_lambda = torch.zeros(self.said_size)
        last_layer_lambda[self.said_size-2:] = torch.ones(2)
        self.lambdas = nn.Parameter(torch.ones(self.said_size)) if is_lambda_param else (last_layer_lambda if is_last_layer_only else torch.ones(self.said_size))
        self.P = projector(self.D, self.d, self.trainable_initparams, self.names)
        

    def to(self, *args, **kwargs):
        self._forward_net[0].to(*args, **kwargs)
        self.trainable_initparams = [
            param.to(*args, **kwargs) for param in self.trainable_initparams
        ]
        return super().to(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.training:
            self._forward_net[0].train()
        else:
            self._forward_net[0].eval()


        if self.global_param == None:
            self.global_param = torch.zeros(self.d, device = self.subspace_params.get_device())

        flat_projected_params = self.P @ (self.subspace_params)
        flat_projected_global = self.global_P @ (self.global_param)

        unflattened_params = unflatten_like(flat_projected_params, self.trainable_initparams)
        unflattened_global = unflatten_like(flat_projected_global, self.trainable_initparams)
        
        iterables = zip(range(self.said_size),self.names, self.trainable_initparams, unflattened_params, unflattened_global)

        for i,p_name, init, proj_param, proj_global in iterables:
            p = self.global_lambdas[i]*(init + proj_global.view(*init.shape)) + (self.lambdas[i] * proj_param.view(*init.shape))
            _setchainattr(self._forward_net[0], p_name, p)
        return self._forward_net[0](*args, **kwargs)

def count_el(list):
    size = 0
    for l in list:
        size += l.numel()
    return size

class LAYEREDModule(nn.Module):

    def __init__(self, net, projector,groups = None, dimension=1000):
        super().__init__()

        self.groups = groups
        self._forward_net = [net]
        initnet = deepcopy(net)
        for orig_name, orig_p in initnet.named_parameters():
            if orig_p.requires_grad:
                _delchainattr(net, orig_name)
                
        aux = [(n, p) for n, p in initnet.named_parameters() if p.requires_grad]

        self.names, self.trainable_initparams = zip(*aux)
        self.trainable_initparams = [param for param in self.trainable_initparams]
        self.names = list(self.names)

        self.layer_size = len(self.groups)
        self.D = sum([param.numel() for param in self.trainable_initparams])
        #self.d_i = np.full(self.layer_size,int((dimension)/self.layer_size) , dtype=int) 
        self.d_i = [int((i*dimension)/np.sum(groups)) for i in groups] #weighted lengths
        self.d = np.sum(self.d_i)
        self.subspace_params = nn.Parameter(torch.zeros(self.d))
        self.indices = np.append(np.cumsum(groups),[0])
        
        self.P = []
        
        self.P = [projector(int(count_el(self.trainable_initparams[self.indices[i-1]:self.indices[i]])),
             int(self.d_i[i]),self.trainable_initparams[self.indices[i-1]:self.indices[i]],self.names[self.indices[i-1]:self.indices[i]]) for i in range(self.layer_size)]
        #print(self.P)
    
    def to(self, *args, **kwargs):
        self._forward_net[0].to(*args, **kwargs)
        self.trainable_initparams = [
            param.to(*args, **kwargs) for param in self.trainable_initparams
        ]
        return super().to(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.training:
            self._forward_net[0].train()
        else:
            self._forward_net[0].eval()


        layer_unflattened_params = []
        d_s = np.append(np.cumsum(self.d_i),[0])
        for i in range(self.layer_size):
            flat_projected_params = self.P[i] @ self.subspace_params[d_s[i-1]:d_s[i]] 
            unflattened_params = unflatten_like(flat_projected_params, self.trainable_initparams[self.indices[i-1]:self.indices[i]])
            for tens in unflattened_params:
                layer_unflattened_params.append(tens)  
            
        iterables = zip(self.names, self.trainable_initparams, layer_unflattened_params)
       
        for p_name, init, proj_param in iterables:
            p = init + proj_param.view(*init.shape)
            _setchainattr(self._forward_net[0], p_name, p)

            
        return self._forward_net[0](*args, **kwargs)



class RandomMultiply(torch.autograd.Function):
    CHUNK_MAX = 2e8

    @staticmethod
    def forward(ctx, v, D, d, seed):
        ctx.info = (D, d, seed)
        with FixedPytorchSeed(seed):
            D_chunks = int(np.ceil((D * d) / RandomMultiply.CHUNK_MAX))
            D_chunksize = D // D_chunks
            D_tot = 0
            Pv_chunks = []
            while D_tot < D:
                D_chunk = min(D_chunksize, D - D_tot)
                D_tot += D_chunk
                Pv_chunks.append(
                    torch.randn(D_chunk, d, device=v.device) @ v / np.sqrt(D)
                )
            Pv = torch.cat(Pv_chunks, dim=0)
        return Pv

    @staticmethod
    def backward(ctx, grad_output):

        D, d, seed = ctx.info
        grad_in = 0.0
        with FixedPytorchSeed(seed):
            D_chunks = int(np.ceil((D * d) / RandomMultiply.CHUNK_MAX))
            D_chunksize = D // D_chunks
            split_grad_outs = torch.split(grad_output, D_chunksize, dim=0)
            for grad_out in split_grad_outs:
                grad_in += (
                    torch.randn(grad_out.shape[0], d, device=grad_output.device).T
                    @ grad_out
                    / np.sqrt(D)
                )
        return grad_in, None, None, None


class LazyRandom(LinearOperator):
    def __init__(self, D, d, params, names, seed=_DEFAULT_SEED):
        super().__init__(None, (D, d))
        self.info = (D, d, seed)

    def _matvec(self, v):
        D, d, seed = self.info
        return RandomMultiply.apply(v, *self.info)

    def _matmat(self, v):
        D, d, seed = self.info
        return RandomMultiply.apply(v, *self.info)

    def __repr__(self):
        return f"LazyRandom({self.D}, {self.d}, seed={self.seed})"


class LazyRandomQR(LinearOperator):
    def __init__(self, D, d, params, names, seed=_DEFAULT_SEED):
        super().__init__(None, (D, d))
        self.info = (D, d, seed)
        self.P = torch.randn(D, d)
        self.P, _ = torch.linalg.qr(self.P, mode="reduced")

    def _matvec(self, v):
        return self.P.to(v.device) @ v

    def _matmat(self, v):
        return self.P.to(v.device) @ v

    def __repr__(self):
        return f"LazyRandomQR({self.D}, {self.d}, seed={self.seed})"


class LazyOneSidedKron(LinearOperator):
    def __init__(self, D, d, params, names, order=2, seed=_DEFAULT_SEED):
        super().__init__(None, (D, d))
        self.seed = seed
        assert np.floor(D ** (1 / order)) == D ** (1 / order)
        self.order = order

    def _matvec(self, v):
        seed = self.seed
        k = int(self.shape[0] ** (1 / self.order))
        out_tensor = torch.zeros(*(self.order * [k]), device=v.device)
        for i in range(self.order):
            Pvi = RandomMultiply.apply(v, k, self.shape[-1], seed) / (
                np.sqrt(self.order) * np.sqrt(k) ** (self.order - 1)
            )
            # unsqueeze all axes except i
            for j in range(self.order):
                if j != i:
                    Pvi = Pvi.unsqueeze(j)
            out_tensor += Pvi
            # re randomize/ advance the seed
            with FixedPytorchSeed(seed):
                seed = int(torch.randint(high=2**31, size=(1,))[0])
        return out_tensor


def RoundedKron(D, d, params, names, order=2, seed=_DEFAULT_SEED):
    rounded_D = int(np.floor(D ** (1 / order))) ** order
    with FixedPytorchSeed(seed):
        fitting_kron = LazyOneSidedKron(rounded_D, d, params, names, order, seed)
        perm = torch.randperm(D)
        if rounded_D == D:
            return LazyPerm(perm) @ fitting_kron
        else:
            newseed = int(torch.randint(high=2**31, size=(1,))[0])
            leftover_random = LazyRandom(D - rounded_D, d, params, names, newseed) * (
                1 / np.sqrt(D / (D - rounded_D))
            )
            return LazyPerm(perm) @ ConcatLazy([fitting_kron, leftover_random])


def RoundedDoubleKron(D, d, params, names, order=2, seed=_DEFAULT_SEED):
    rounded_D = int(np.floor(D ** (1 / order)))
    rounded_d = int(np.floor(d ** (1 / order)))

    with FixedPytorchSeed(seed):
        seed = int(torch.randint(high=2**31, size=(1,))[0])
        Rs = []
        for i in range(order):
            Rs.append(LazyRandom(rounded_D, rounded_d, params, names, seed))
            seed = int(torch.randint(high=2**31, size=(1,))[0])
        RkR = LazyKron(Rs)
        if rounded_D**order == D or rounded_d**order == d:
            extra = Lazy(
                torch.randn(D - rounded_D**order, d - rounded_d**order) / np.sqrt(D)
            )
        else:
            extra = LazyRandom(
                D - rounded_D**order, d - rounded_d**order, params, names, seed
            )

        M = LazyDirectSum([RkR, extra])
        perm = torch.randperm(D)

    return LazyPerm(perm) @ M


def RoundedDoubleKronQR(D, d, params, names, order=2, seed=_DEFAULT_SEED):
    rounded_D = int(np.floor(D ** (1 / order)))
    rounded_d = int(np.floor(d ** (1 / order)))
    # TODO: double check that there is no normalization as the QR does not need it

    with FixedPytorchSeed(seed):
        seed = int(torch.randint(high=2**31, size=(1,))[0])
        Rs = []
        for i in range(order):
            Rs.append(LazyRandomQR(rounded_D, rounded_d, params, names, seed))
            seed = int(torch.randint(high=2**31, size=(1,))[0])
        RkR = LazyKron(Rs)
        if rounded_D**order == D or rounded_d**order == d:
            extra = Lazy(
                torch.randn(D - rounded_D**order, d - rounded_d**order) / np.sqrt(D)
            )
        else:
            extra = LazyRandom(
                D - rounded_D**order, d - rounded_d**order, params, names, seed
            )

        M = LazyDirectSum([RkR, extra])
        perm = torch.randperm(D)

    return LazyPerm(perm) @ M


def FiLMLazyRandom(D, d, params, names, seed=_DEFAULT_SEED):
    def bn_or_fc(name):
        return (
            ("bn" in name)
            or ("fc" in name)
            or ("norm" in name)
            or ("classifier" in name)
        )

    return FilterLazyRandom(D, d, params, names, bn_or_fc, seed)


class FilterLazyRandom(LinearOperator):
    def __init__(self, D, d, params, names, condition, seed=_DEFAULT_SEED):
        super().__init__(None, (D, d))
        i = 0
        ids = []
        for name, param in zip(names, params):
            if condition(name):
                ids.append(np.arange(i, i + param.numel()))
            i += param.numel()
        self.ids = np.concatenate(ids)
        assert len(ids) > 0
        assert i == D
        self.dense_random = LazyRandom(len(self.ids), d, params, names, seed)
        #print(D, len(self.ids), d)

    def _matvec(self, v):
        filtered_v_params = self.dense_random @ v
        out = torch.zeros(self.shape[0], device=v.device, dtype=v.dtype)
        out[self.ids] = filtered_v_params
        return out


class LazySTFiLMRDKronQR(LinearOperator):
    def __init__(self, D, d, params, names, seed=_DEFAULT_SEED):
        super().__init__(None, (D, d))
        def condition_fn1(x): return x.find('bn') >= 0
        def condition_fn2(x): return not condition_fn1(x)
        ids1 = find_locations_from_condition(names, params, condition_fn1)
        ids2 = find_locations_from_condition(names, params, condition_fn2)
        self.bn_d = len(ids1)
        self.ids = np.argsort(np.concatenate([ids1, ids2]))
        self.P = RoundedDoubleKronQR(D - self.bn_d, d - self.bn_d, params, names)

    def _matvec(self, v):
        return self._matmat(v)

    def _matmat(self, v):
        v1, v2 = v[:self.bn_d], v[self.bn_d:]
        output = torch.concat([v1, self.P @ v2])
        return output[self.ids]


def find_locations_from_condition(names, params, condition_fn):
    i, ids = 0, []
    for name, param in zip(names, params):
        if condition_fn(name):
            ids.append(np.arange(i, i + param.numel()))
        i += param.numel()
    ids = np.concatenate(ids)
    return ids


def find_all_batch_norm(net):
    leaf_criteria = (nn.BatchNorm1d, nn.BatchNorm2d)

    class Counter:
        count = 0

        def count_params_in_module(self, x):
            print(x)
            for y in list(x.parameters()):
                self.count += y.numel()

    counter = Counter()
    # TODO: check if this is the correct way to pass modules
    selective_apply(list(net.modules())[0], counter, leaf_criteria)
    return counter.count


def is_leaf(module, leaf_criteria):
    no_children_att = not hasattr(module, 'children')
    no_children = not list(module.children())
    is_leaf_criteria = isinstance(module, leaf_criteria)
    return no_children_att or no_children or is_leaf_criteria


def selective_apply(module, counter, leaf_criteria):
    if is_leaf(module, leaf_criteria):
        if isinstance(module, leaf_criteria):
            counter.count_params_in_module(module)
    else:
        for c in module.children():
            selective_apply(c, counter, leaf_criteria)


def check_layer_name(names):
    for name in names:
        if ("bn" in name) or ("fc" in name) or ("norm" in name) or ("classifier" in name):
            return True
    return False

def CombinedRDKronFiLM(D, d, params, names, seed=_DEFAULT_SEED,is_layered = False):
    rdkron = RoundedDoubleKron(D, d, params, names, seed=seed)
    if is_layered and (not check_layer_name(names)):
        return rdkron * (1 / np.sqrt(2))
    FiLM = FiLMLazyRandom(D, d, params, names, seed=seed)
    return (rdkron + FiLM) * (1 / np.sqrt(2))


def CombinedRDKronQRFiLM(D, d, params, names, seed=_DEFAULT_SEED,is_layerd = False):
    rdkronqr = RoundedDoubleKronQR(D, d, params, names, seed=seed)
    if is_layered and (not check_layer_name(names)):
        return rdkronqr * (1 / np.sqrt(2))
        
    FiLM = FiLMLazyRandom(D, d, params, names, seed=seed)
    return (rdkronqr + FiLM) * (1 / np.sqrt(2))


class SparseOperator(LinearOperator):
    def __init__(self, D, d, params, names, seed=_DEFAULT_SEED):
        super().__init__(None, (D, d))
        s = np.sqrt(D)
        with FixedNumpySeed(seed):
            number_nonzero = np.random.binomial(D * d, 1.0 / s)
            # print(number_nonzero)
            nonzero_indices = np.random.choice(D * d, number_nonzero)
            nonzero_indices2d = np.stack(
                np.unravel_index(nonzero_indices, (D, d)), axis=0
            )
            # sample values from +-1
            nonzero_values = np.random.choice([-1, 1], number_nonzero) / np.sqrt(s)
            self.V = torch.sparse_coo_tensor(
                nonzero_indices2d, nonzero_values, size=(D, d)
            ).float()

    def _matvec(self, x):
        assert x.shape[0] == self.shape[-1], f"{x.shape[0]} != {self.shape[-1]}"
        return self.V.to(x.device) @ x


class FastfoodOperator(LinearOperator):
    # Source: https://discuss.pytorch.org/t/fast-walsh-hadamard-transform/19341
    class FWHT(torch.autograd.Function):
        @staticmethod
        def transform(x):
            bit = dd = x.size(-1)
            result = x.detach().cpu().numpy()

            for _ in range(int(np.log2(dd))):
                bit >>= 1
                for i in range(dd):
                    if i & bit == 0:
                        j = i | bit
                        temp = np.copy(result[..., i])
                        result[..., i] += result[..., j]
                        result[..., j] = temp - result[..., j]

            result /= np.sqrt(dd)
            return torch.from_numpy(result).to(x.device)

        @staticmethod
        def forward(_, inputs):
            return FastfoodOperator.FWHT.transform(inputs)

        @staticmethod
        def backward(_, grad_outputs):
            return FastfoodOperator.FWHT.transform(grad_outputs)

    def __init__(self, D, d, params, names, scale=1, seed=_DEFAULT_SEED):
        super().__init__(None, (D, d))

        self.D = D
        self.real_d = d
        self.d = 2 ** np.ceil(np.log2(d)).astype(int)
        self.sigma = scale
        blocks = np.ceil(self.D / self.d).astype(int)
        with FixedPytorchSeed(seed):
            self.S = torch.rand(blocks, self.d)
            self.G = torch.randn(blocks, self.d)
            self.B = 2 * (torch.rand(blocks, self.d) > 0.5).float() - 1
            self.Pi = torch.randperm(self.d)

    def _matvec(self, x):
        """Implicit P @ x
        Assumed x is 1-D tensor.
        """
        device = x.device

        pad = self.d - self.real_d
        if pad > 0:
            x = torch.cat([x, torch.zeros(pad, device=device)], dim=-1)

        GPiHBx = (
            self.G.to(device)
            * FastfoodOperator.FWHT.apply(self.B.to(device) * x)[..., self.Pi]
        )
        SHGPiHBx = self.S.to(device) * FastfoodOperator.FWHT.apply(GPiHBx)
        result = SHGPiHBx.flatten()[: self.D] / (self.sigma * np.sqrt(self.d))
        return result


def create_intrinsic_model(
    base_net,
    is_said = False,
    ckpt_path=None,
    intrinsic_mode="filmrdkron",
    intrinsic_dim=1000,
    seed=None,
    device=None,
    is_layered = False,
    groups = None,
    global_param = None,
    global_P = None,
    is_said_meta = False,
    is_lambda_param = True,
    is_last_layer_only = False
):
    if seed is None:
        raise ValueError(
            "Missing seed. Randomized projections will not be reproducible!"
        )

    #TODO : complete ,global_param = global_param for other intrinsic modes
    net = None
    module = MetaSAIDModule if is_said_meta else( LAYEREDModule if is_layered else (SAIDModule if is_said else IDModule))

    if intrinsic_mode == "dense":
        class DenseIDNet(module):
            def __init__(self, net, dimension=1000,groups = groups, seed=None,global_param = None, **_):
                super().__init__(
                    net, partial(LazyRandom, seed=seed), dimension=dimension ,groups = groups,global_param = global_param
                )
        net = DenseIDNet(base_net, dimension=intrinsic_dim,groups = groups, seed=seed,global_param = global_param)

    elif intrinsic_mode == "sparse":
        class SparseIDNet(module):
            def __init__(self, net, dimension=1000,groups = groups, seed=None,global_param = global_param, **_):
                super().__init__(
                    net, partial(SparseOperator, seed=seed), dimension=dimension,groups = groups,global_param = global_param
                )
        net = SparseIDNet(base_net, dimension=intrinsic_dim,groups = groups, seed=seed,global_param = global_param)

    elif intrinsic_mode == "fastfood":
        class FastfoodIDNet(module):
            def __init__(self, net, dimension=1000,groups = groups, seed=None, **_):
                super().__init__(
                    net, partial(FastfoodOperator, seed=seed), dimension=dimension ,groups = groups
                )
        net = FastfoodIDNet(base_net, dimension=intrinsic_dim,groups = groups, seed=seed)

    elif intrinsic_mode == "rkron":
        class RoundedKronIDNet(module):
            def __init__(self, net, dimension=1000,groups = groups, order=2, seed=None, **_):
                super().__init__(
                    net,
                    partial(RoundedKron, order=order, seed=seed),dimension=dimension,groups = groups
                )
        net = RoundedKronIDNet(base_net, dimension=intrinsic_dim,groups = groups, seed=seed
                              )

    elif intrinsic_mode == "rdkron":
        class RoundedDoubleKronIDNet(module):
            def __init__(self, net, dimension=1000,groups = groups, order=2, seed=None, **_):
                super().__init__(
                    net,
                    partial(RoundedDoubleKron, order=order, seed=seed),dimension=dimension,groups = groups
                )
        net = RoundedDoubleKronIDNet(base_net, dimension=intrinsic_dim,groups = groups, seed=seed 
                                    )

    elif intrinsic_mode == "rdkronqr":
        class RoundedDoubleKronQRIDNet(module):
            def __init__(self, net, dimension=1000,groups = groups, order=2, seed=None, **_):
                super().__init__(
                    net,
                    partial(RoundedDoubleKronQR, order=order, seed=seed),dimension=dimension,groups = groups
                )
        net = RoundedDoubleKronQRIDNet(base_net, dimension=intrinsic_dim,groups = groups, seed=seed
                                      )

    elif intrinsic_mode == "film":
        class FiLMIDNet(module):
            def __init__(self, net, dimension=1000,groups = groups, seed=None, **_):
                super().__init__(
                    net, partial(FiLMLazyRandom, seed=seed), dimension=dimension ,groups = groups
                )
        net = FiLMIDNet(base_net, dimension=intrinsic_dim,groups = groups, seed=seed
                       )

    elif intrinsic_mode == "filmrdkron":
        class FiLMRDKronIDNet(module):
            def __init__(self, net, dimension=1000,groups = None, seed=None,global_param = global_param, global_P = global_P, is_lambda_param=is_lambda_param, is_last_layer_only=is_last_layer_only, **_):
                super().__init__(
                    net, partial(CombinedRDKronFiLM, seed=seed,is_layered = is_layered), dimension=dimension ,groups = groups,global_param = global_param, global_P = global_P, is_lambda_param = is_lambda_param, is_last_layer_only=is_last_layer_only
                )
        net = FiLMRDKronIDNet(base_net, dimension=intrinsic_dim,groups = groups, seed=seed,global_param = global_param, global_P = global_P, is_lambda_param = is_lambda_param, is_last_layer_only=is_last_layer_only
                             )

    elif intrinsic_mode == "filmrdkronqr":
        class FiLMRDKronQRIDNet(module):
            def __init__(self, net, dimension=1000,groups = groups, seed=None, **_):
                super().__init__(
                    net, partial(CombinedRDKronQRFiLM, seed=seed ,is_layered = is_layered), dimension=dimension,groups = groups
                )
        net = FiLMRDKronQRIDNet(base_net, dimension=intrinsic_dim,groups = groups, seed=seed
                               )

    elif intrinsic_mode == "stfilmkronqr":
        class STFiLMRDKronQRIDNet(module):
            def __init__(self, net, dimension=1000,groups = groups, seed=None, **_):
                super().__init__(
                    net, partial(LazySTFiLMRDKronQR, seed=seed), dimension=dimension,groups = groups
                )
        net = STFiLMRDKronQRIDNet(base_net, dimension=intrinsic_dim,groups = groups, seed=seed
                                 )

    else:
        raise NotImplementedError

    if ckpt_path is not None:
        weights = torch.load(ckpt_path)
        if "subspace_params" in weights:
            print(len(weights["subspace_params"]))
            print(len(net.subspace_params))
            net.load_state_dict(weights)
        else:
            tmp = {}
            tmp["subspace_params"] = weights["module.subspace_params"]
            net.load_state_dict(tmp)
    return net
