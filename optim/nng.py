from collections import defaultdict, abc as container_abcs
from copy import deepcopy
from itertools import chain
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

import curvature as curvmat
curv_names = [name for name in curvmat.__dict__ if not name.startswith('__') and callable(curvmat.__dict__[name])]

class NoisyNaturalGradient(Optimizer):

    def __init__(self, model: nn.Module, dataset_size: int, curv_shapes: dict, curv_kwargs: dict,
                 lr=0.01, gain=1.0, momentum=0.9, precision=None, ema_decay=0.99, weight_decay=0.0,
                 kl_lam=1., kl_clip=1e-4, mul_factor=1.0, seed=1, cov_T=1, inv_T=1):

        self.model = model
        scale = math.sqrt(kl_lam / dataset_size)

        defaults = {'lr': lr, 'momentum': momentum, 'ema_decay': ema_decay,
                    'weight_decay': weight_decay, 'scale': scale, 'seed_base': seed}
        defaults.update(curv_kwargs)
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.optim_state = {'step': 0}
        self.kl_clip = kl_clip
        self.mul_factor = mul_factor
        self.cov_T = cov_T
        self.inv_T = inv_T

        self.param_groups = []
        self.curv_shapes = {} if curv_shapes is None else curv_shapes

        print("sam_damping : {}".format(precision * kl_lam / dataset_size))
        for module in model.modules():
            if len(list(module.children())) > 0:
                continue
            params = list(module.parameters())
            if len(params) == 0:
                continue

            curv_class = self.get_curv_class(module)
            if curv_class not in curv_names:
                curvature = None
            else:
                if precision == 0.0:
                    precision = np.prod(module.weight.shape[1:]) / gain
                if kl_lam == 0.0 and weight_decay > 0.0:
                    sam_damping = weight_decay
                else:
                    sam_damping = precision * kl_lam / dataset_size
                curvature = curvmat.__dict__[curv_class](module, sam_damping=sam_damping, **curv_kwargs)

            group = {
                'params': params,
                'curv': curvature,
            }

            self.add_param_group(group)
            self.init_buffer(params)

        for group in self.param_groups:
            group['mean'] = [p.data.detach().clone() for p in group['params']]

            if group['curv'] is not None:
                curv = group['curv']
                curv.init(1.0)
                curv.update(group['ema_decay'], update_est=False, update_inv_cov=True)

    def get_curv_class(self, module):
        module_name = module.__class__.__name__
        curv_shape = self.curv_shapes.get(module_name, '')
        curv_name = curv_shape + module_name
        return curv_name

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for m in group['mean']:
                if m.grad is not None:
                    m.grad.detach_()
                    m.grad.zero_()
            if group['curv'] is not None:
                for data in group['curv'].data:
                    data.zero_()

        super(NoisyNaturalGradient, self).zero_grad()

    def init_buffer(self, params):
        for p in params:
            state = self.state[p]
            state['momentum_buffer'] = torch.zeros_like(p.data)

    @property
    def seed(self):
        return self.optim_state['step'] + self.defaults['seed_base']

    def set_random_seed(self, seed=None):
        if seed is None:
            seed = self.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def sample_params(self, mul_factor=1.0):
        for group in self.param_groups:
            params, mean = group['params'], group['mean']
            curv = group['curv']
            if curv is not None and group['scale'] > 0.0 and mul_factor > 0.0:
                # sample from posterior
                curv.sample_params(params, mean, group['scale'] * mul_factor)
            else:
                for p, m in zip(params, mean):
                    p.data.copy_(m.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        def closure():
            # forward/backward
            return loss
        """

        self.set_random_seed()

        # sampling (warmup with 100 steps)
        self.sample_params(self.mul_factor)

        # forward and backward
        update_est = self.optim_state['step'] % self.cov_T == 0
        update_inv_cov = self.optim_state['step'] % self.inv_T == 0
        loss, output = closure(update_est)
        self.optim_state['step'] += 1

        # compute preconditioned update for mean parameters and also update curvature
        for group in self.param_groups:
            self._preprocess(group)

            # update covariance
            mean, curv = group['mean'], group['curv']
            if curv is not None:
                curv.update(group['ema_decay'], update_est=update_est, update_inv_cov=update_inv_cov)
                curv.preconditioning(mean)

        # kl clipping
        self._kl_clipping()

        # take the update for mean
        for group in self.param_groups:
            # update mean
            self.update(group)

            # copy mean to param
            self.copy_mean_to_params()

        return loss, output

    def copy_mean_to_params(self):
        for group in self.param_groups:
            # copy mean to param
            mean = group['mean']
            params = group['params']
            for p, m in zip(params, mean):
                p.data.copy_(m.data)

    def _preprocess(self, group):
        means = group['mean']
        params = group['params']

        for m, p in zip(means, params):
            m.grad = p.grad.clone()
            if group['curv'] is not None:
                m.grad.data.add_(p.data, alpha=group['curv'].sam_damping)
                p.grad.data.add_(p.data, alpha=group['curv'].sam_damping)
            # else:
            #     m.grad.data.add_(p.data, alpha=group['weight_decay'])
            #     p.grad.data.add_(p.data, alpha=group['weight_decay'])

    def _kl_clipping(self):
        kl_dist = 0
        for group in self.param_groups:
            means = group['mean']
            params = group['params']

            lr = group['lr']
            for m, p in zip(means, params):
                kl_dist += (lr ** 2 * m.grad.data * p.grad.data).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / kl_dist))

        for group in self.param_groups:
            means = group['mean']
            for m in means:
                m.grad.data.mul_(nu)

    def update(self, group):
        def apply_momentum(p, grad):
            momentum = group['momentum']
            if momentum != 0:
                buf = self.state[p]['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                grad = buf
            return grad

        params = group['params']
        means = group['mean']
        for m, p in zip(means, params):
            grad = m.grad
            if grad is None:
                continue
            new_grad = apply_momentum(p, grad) # momentum_buffer is under params
            m.data.add_(new_grad, alpha=-group['lr'])

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if (k != 'params' and k != 'mean')}
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                   if id(p) not in param_mappings})
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain.from_iterable((g['params'] for g in saved_groups)),
                      chain.from_iterable((g['params'] for g in groups)))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']

            # TODO (GD): the following is a bit hacky, figure out a better way
            if group['curv'] is not None:
                new_group['curv'].set_module(group['curv'].module)
                new_group['curv'].register_hook()
            return new_group

        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

        # copy to mean
        for group in self.param_groups:
            group['mean'] = [p.data.detach().clone() for p in group['params']]


class NKFAC(NoisyNaturalGradient):

    def __init__(self, *args, **kwargs):
        default_kwargs = dict(lr=1e-3,
                              curv_shapes={
                                  'Linear': 'Kron',
                                  'Conv2d': 'Kron',
                              },
                              curv_kwargs={'damping': 1e-3},
                              ema_decay=0.99,
                              cov_T=10,
                              inv_T=100,
                              momentum=0.9)

        default_kwargs.update(kwargs)

        super(NKFAC, self).__init__(*args, **default_kwargs)


class NAdam(NoisyNaturalGradient):

    def __init__(self, *args, **kwargs):
        default_kwargs = dict(lr=1e-3,
                              curv_shapes={
                                  'Linear': 'Diag',
                                  'Conv2d': 'Diag',
                              },
                              curv_kwargs={'damping': 1e-3},
                              ema_decay=0.99,
                              cov_T=1,
                              inv_T=1,
                              momentum=0.9)

        default_kwargs.update(kwargs)

        super(NAdam, self).__init__(*args, **default_kwargs)
