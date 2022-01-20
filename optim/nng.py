from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

import curvature as curvmat
curv_names = [name for name in curvmat.__dict__ if not name.startswith('__') and callable(curvmat.__dict__[name])]

class NoisyNaturalGradient(Optimizer):

    def __init__(self, model: nn.Module, dataset_size: int, curv_shapes: dict, curv_kwargs: dict,
                 lr=0.01, momentum=0.9, precision=20., kl_lam=1., seed=1, inv_T=1, kl_clip=1e-4):

        self.model = model
        sam_damping = precision * kl_lam / dataset_size
        scale = math.sqrt(kl_lam / dataset_size)

        defaults = {'lr': lr, 'momentum': momentum, 'scale': scale, 'seed_base': seed}
        curv_kwargs['sam_damping'] = sam_damping
        defaults.update(curv_kwargs)
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.optim_state = {'step': 0}
        self.kl_clip = kl_clip
        self.inv_T = inv_T

        self.param_groups = []
        self.curv_shapes = {} if curv_shapes is None else curv_shapes

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
                curvature = curvmat.__dict__[curv_class](module, **curv_kwargs)

            group = {
                'params': params,
                'curv': curvature,
            }

            self.add_param_group(group)
            self.init_buffer(params)

        for group in self.param_groups:
            group['mean'] = [p.data.detach().clone() for p in group['params']]
            self.init_buffer(group['mean'])

            if group['curv'] is not None:
                curv = group['curv']
                curv.init(1.0)
                curv.update(update_est=False, update_inv_cov=True)

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

    def sample_params(self):
        for group in self.param_groups:
            params, mean = group['params'], group['mean']
            curv = group['curv']
            if curv is not None:
                # sample from posterior
                curv.sample_params(params, mean, group['scale'])
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
            return loss, output
        """

        self.set_random_seed()

        # sampling
        self.sample_params()

        # forward and backward
        loss, output = closure()
        self.optim_state['step'] += 1

        # compute preconditioned update for mean parameters and also update curvature
        for group in self.param_groups:
            self._preprocess(group)

            # update covariance
            mean, curv = group['mean'], group['curv']
            if curv is not None:
                curv.update(update_inv_cov=(self.optim_state['step'] % self.inv_T == 1))
                curv.preconditioning(mean)

        # kl clipping
        self._kl_clipping()

        # take the update for mean
        for group in self.param_groups:
            # update mean
            self.update(group, target='mean')

            # copy mean to param
            mean = group['mean']
            params = group['params']
            for p, m in zip(params, mean):
                p.data.copy_(m.data)
                p.grad.copy_(m.grad)

        return loss

    def _preprocess(self, group):
        means = group['mean']
        params = group['params']

        for m, p in zip(means, params):
            m.grad = p.grad.clone()
            m.grad.data.add_(group['sam_damping'], p.data)

    def _kl_clipping(self):
        kl_dist = 0
        for group in self.param_groups:
            means = group['mean']
            params = group['params']

            lr = group['lr']
            for m, p in zip(means, params):
                kl_dist += (lr ** 2 * m.grad.data * (p.grad.data + group['sam_damping'] * p.data)).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / kl_dist))

        for group in self.param_groups:
            means = group['mean']
            for m in means:
                m.grad.data.mul_(nu)

    def update(self, group, target='params'):
        def apply_momentum(p, grad):
            momentum = group['momentum']
            if momentum != 0:
                buf = self.state[p]['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                grad = buf
            return grad

        params = group[target]
        for p in params:
            grad = p.grad
            if grad is None:
                continue
            new_grad = apply_momentum(p, grad)
            p.data.add_(-group['lr'], new_grad)


class NKFAC(NoisyNaturalGradient):

    def __init__(self, *args, **kwargs):
        default_kwargs = dict(lr=1e-3,
                              curv_shapes={
                                  'Linear': 'Kron',
                                  'Conv2d': 'Kron',
                              },
                              curv_kwargs={'ema_decay': 0.98, 'damping': 1e-3},
                              inv_T=20,
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
                              curv_kwargs={'ema_decay': 0.98, 'damping': 1e-3},
                              inv_T=1,
                              momentum=0.9)

        default_kwargs.update(kwargs)

        super(NAdam, self).__init__(*args, **default_kwargs)
