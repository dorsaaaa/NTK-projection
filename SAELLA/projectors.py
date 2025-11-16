import sys
import os
project_dir = os.getcwd()
sys.path.append(project_dir)

import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
import random
from functools import partial
from pactl.nn.linear_operator_base import (
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

    def __init__(self, net, basis_num, dimension=1000, global_param = None, global_lambdas=None, global_P = None, is_lambda_global=True, is_base_different=False):
        super().__init__()

        self.basis_num = basis_num
        self.global_param = global_param
        self.global_lambdas = global_lambdas
        self.global_P = global_P
        self._forward_net = [net]
        initnet = deepcopy(net)
        self.said_size = len([p for n,p in net.named_parameters() if p.requires_grad])
        self.d = dimension - self.said_size
        self.lambdas = None if (is_lambda_global or is_base_different) else nn.Parameter(torch.ones(self.said_size)/self.said_size, requires_grad = True)
        
        self.is_lambda_global = is_lambda_global
        self.is_base_different = is_base_different
        self.use_softmax=False

        for orig_name, orig_p in initnet.named_parameters():
            if orig_p.requires_grad:
                _delchainattr(net, orig_name)
                
        aux = [(n, p) for n, p in initnet.named_parameters() if p.requires_grad]
        
        self.names, self.trainable_initparams = zip(*aux)
        self.trainable_initparams = [param for param in self.trainable_initparams]
        self.names = list(self.names)
      
        self.subspace_params = nn.Parameter(torch.ones(self.basis_num)/self.basis_num, requires_grad = True)
        


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

        lambdas = self.global_lambdas if self.is_lambda_global else self.lambdas
        lambdas = torch.softmax(lambdas, dim=0) if self.use_softmax else lambdas
        
        if self.is_base_different:
            aw_0 =self.subspace_params[0] * self.global_param[:self.d]
            l_0 = lambdas[:self.said_size]
            aPw_0= self.global_P[0] @ aw_0
            unflattened_params = unflatten_like(aPw_0, self.trainable_initparams)
            for j in range(self.said_size):
                unflattened_params[j] *= l_0[j]

            for i in range(1,self.basis_num):
                aw =self.subspace_params[i] * self.global_param[i*self.d:(i+1)*self.d]
                l_i = lambdas[i*self.said_size:(i+1)*self.said_size]
                aPw= self.global_P[i] @ aw
                unflattened_params_i = unflatten_like(aPw, self.trainable_initparams)
                for j in range(self.said_size):
                    unflattened_params_i[j] *= l_i[j]
                    unflattened_params[j]+= unflattened_params_i[j]

            iterables = zip(self.names, self.trainable_initparams, unflattened_params)
            for p_name, init, proj_param in iterables:
                p =init + proj_param.view(*init.shape)
                _setchainattr(self._forward_net[0], p_name, p)

        else:
            coefficients = torch.repeat_interleave(self.subspace_params, self.d).to(self.subspace_params.device)
            weights = coefficients * self.global_param

            flat_projected_params = self.global_P @ weights


            unflattened_params = unflatten_like(
                flat_projected_params, self.trainable_initparams
            )
            iterables = zip(lambdas,self.names, self.trainable_initparams, unflattened_params)

            for lam,p_name, init, proj_param in iterables:
                p =init + lam*proj_param.view(*init.shape)
                _setchainattr(self._forward_net[0], p_name, p)

        
        return self._forward_net[0](*args, **kwargs)







def count_el(list):
    size = 0
    for l in list:
        size += l.numel()
    return size



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
        print(D, len(self.ids), d)

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
    ckpt_path=None,
    intrinsic_mode="filmrdkron",
    basis_num = None,
    intrinsic_dim=1000,
    seed=None,
    device=None,
    global_param = None,
    global_lambdas=None,
    global_P = None,
    is_lambda_global = True,
    is_base_different = False
):
    if seed is None:
        raise ValueError(
            "Missing seed. Randomized projections will not be reproducible!"
        )

    module = IDModule

    if intrinsic_mode == "filmrdkron":
        class FiLMRDKronIDNet(module):
            def __init__(self, net, basis_num = basis_num, dimension=1000, seed=None,global_param = global_param, global_lambdas=global_lambdas, global_P = global_P, is_lambda_global=is_lambda_global, is_base_different=is_base_different, **_):
                super().__init__(
                    net, dimension=dimension , basis_num = basis_num,global_param = global_param, global_lambdas=global_lambdas, global_P = global_P, is_lambda_global=is_lambda_global,  is_base_different=is_base_different
                )
        net = FiLMRDKronIDNet(base_net, basis_num = basis_num, dimension=intrinsic_dim, seed=seed,global_param = global_param, global_lambdas=global_lambdas, global_P = global_P, is_lambda_global=is_lambda_global,  is_base_different=is_base_different
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
