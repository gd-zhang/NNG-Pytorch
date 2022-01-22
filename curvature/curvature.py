import math
import torch
import torch.cuda.comm
from utils.kfac_utils import update_running_stats


class Curvature(object):
    def __init__(self, module, ema_decay=1.0, damping=1e-3, sam_damping=1e-3):
        assert 0.0 <= ema_decay <= 1.0, "Invalid ema_decay value"

        self._module = module
        self._ema_decay = ema_decay
        self._damping = max(damping, sam_damping)
        self._sam_damping = sam_damping

        self._acc_stats = False
        self._data = None
        self._est = None

        module.register_forward_hook(self.forward_process)
        module.register_backward_hook(self.backward_process)

    def set_acc_stats(self, value):
        self._acc_stats = value

    def _get_shape(self):
        NotImplementedError

    @property
    def shape(self):
        return self._get_shape()

    @property
    def data(self):
        return self._data

    @property
    def sam_damping(self):
        return self._sam_damping

    @property
    def device(self):
        return next(self._module.parameters()).device

    @property
    def module(self):
        return self._module

    @property
    def bias(self):
        bias = getattr(self._module, 'bias', None)
        return False if bias is None else True

    def forward_process(self, module, input, output):
        if self._acc_stats:
            data_input = input[0].detach()
            # setattr(module, "data_input", data_input)
            self.update_forward(data_input)

    def backward_process(self, module, grad_input, grad_output):
        if self._acc_stats:
            grad_output = grad_output[0].data
            # setattr(module, "grad_output", grad_output)
            self.update_backward(grad_output)

    def update_forward(self, data_input):
        NotImplementedError

    def update_backward(self, grad_output):
        NotImplementedError

    def _compute_curv(self):
        NotImplementedError

    def update(self, update_est=True, update_inv_cov=False):
        if update_est:
            self._compute_curv()
            for est, data in zip(self._est, self._data):
                update_running_stats(data, est, self._ema_decay)

        if update_inv_cov:
            self._update_inv_cov()

    def _update_inv_cov(self):
        NotImplementedError

    def sample_params(self, params, means, scale):
        NotImplementedError

    def preconditioning(self, params):
        NotImplementedError


class DiagCurvature(Curvature):
    def __init__(self, *args, **kwargs):
        super(DiagCurvature, self).__init__(*args, **kwargs)

        self._inv = None
        self._cov = None
        self._data_inputs = []
        self._grad_outputs = []

    def _get_shape(self):
        return [p.shape for p in self._module.parameters()]

    def update_forward(self, data_input):
        self._data_inputs.append(data_input)

    def update_backward(self, grad_output):
        self._grad_outputs.append(grad_output)

    def init(self, value):
        self._est = [torch.ones(s, device=self.device).mul(value) for s in self.shape]
        self._data = [torch.zeros(s, device=self.device).mul(value) for s in self.shape]

    def _update_inv_cov(self):
        self._inv = [1.0 / (est + self._damping * torch.ones_like(est, device=self.device)) for est in self._est]
        self._cov = [1.0 / (est + self._sam_damping * torch.ones_like(est, device=self.device)) for est in self._est]

    def preconditioning(self, params):
        for param, inv in zip(params, self._inv):
            param.grad.copy_(inv.mul(param.grad))

    def sample_params(self, params, means, scale):
        for param, mean, cov in zip(params, means, self._cov):
            noise = torch.randn_like(mean)
            param.data.copy_(mean.add(noise * cov.sqrt() * scale))


class KronCurvature(Curvature):
    def __init__(self, *args, **kwargs):
        super(KronCurvature, self).__init__(*args, **kwargs)

        self._eigval = None
        self._eigvec = None

    def init(self, value):
        self._est = [torch.eye(s[0], device=self.device).mul(value) for s in self.shape]
        self._data = [torch.zeros(s, device=self.device).mul(value) for s in self.shape]
        self._A = []
        self._G = []

    def _compute_curv(self):
        # average over all devices
        self._data[0].copy_(torch.cuda.comm.reduce_add(self._A)).div_(len(self._A))
        # TODO (GD): double check the following
        self._data[1].copy_(torch.cuda.comm.reduce_add(self._G)).mul_(len(self._G))
        self._A = []
        self._G = []

    def _get_grad_mat(self):
        NotImplementedError

    def _update_inv_cov(self):
        self._eigval = []
        self._eigvec = []
        for est in self._est:
            e, v = torch.linalg.eigh(est)
            self._eigval.append(e.clamp(min=0.0))
            self._eigvec.append(v)

    def preconditioning(self, params):
        pgrad_mat = self._get_grad_mat()
        pgrad_mat = self._eigvec[1].t() @ pgrad_mat @ self._eigvec[0]
        pgrad_mat = pgrad_mat / (self._eigval[1].unsqueeze(1) * self._eigval[0].unsqueeze(0) + self._damping)
        pgrad_mat = self._eigvec[1] @ pgrad_mat @ self._eigvec[0].t()

        if not self.bias:
            params[0].grad.copy_(pgrad_mat.view(params[0].grad.data.size()))
        else:
            pgrad_mat = [pgrad_mat[:, :-1], pgrad_mat[:, -1:]]
            params[0].grad.copy_(pgrad_mat[0].view(params[0].grad.data.size()))
            params[1].grad.copy_(pgrad_mat[1].view(params[1].grad.data.size()))

    def sample_params(self, params, means, scale):
        noise = torch.randn([self._eigvec[1].shape[0], self._eigvec[0].shape[0]], device=self.device)
        # noise = self._eigvec[1].t() @ noise @ self._eigvec[0] # this is not necessary for sampling
        noise = noise / (self._eigval[1].unsqueeze(1) * self._eigval[0].unsqueeze(0) + self._sam_damping).sqrt()
        noise = self._eigvec[1] @ noise @ self._eigvec[0].t()
        noise = noise * scale

        if not self.bias:
            params[0].data.copy_(means[0].data + noise.view(params[0].data.size()))
        else:
            noise = [noise[:, :-1], noise[:, -1:]]
            params[0].data.copy_(means[0].data + noise[0].view(params[0].data.size()))
            params[1].data.copy_(means[1].data + noise[1].view(params[1].data.size()))










