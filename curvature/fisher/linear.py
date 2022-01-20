import torch
from curvature.curvature import DiagCurvature, KronCurvature


class DiagLinear(DiagCurvature):

    def _compute_curv(self):
        data_input = torch.cuda.comm.gather(self._data_inputs, dim=0)
        grad_output = torch.cuda.comm.gather(self._grad_outputs, dim=0)
        batch_size = data_input.size(0)
        grad_output *= batch_size
        self._data[0].copy_(grad_output.pow(2).t() @ data_input.pow(2).div(batch_size))
        if self.bias:
            self._data[1].copy_(grad_output.pow(2).mean(dim=0))
        self._data_inputs = []
        self._grad_outputs = []


class KronLinear(KronCurvature):

    def _get_shape(self):
        out_dim, in_dim = self._module.weight.data.size()
        if self.bias:
            return [(in_dim+1, in_dim+1), (out_dim, out_dim)]
        else:
            return [(in_dim, in_dim), (out_dim, out_dim)]

    def _get_grad_mat(self):
        p_grad_mat = self._module.weight.grad.data
        if self.bias:
            p_grad_mat = torch.cat([p_grad_mat, self._module.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def update_forward(self, data_input):
        if self.bias is not None:
            data_input = torch.cat([data_input, data_input.new(data_input.size(0), 1).fill_(1)], 1)
        batch_size = data_input.size(0)
        self._A.append(data_input.t() @ (data_input / batch_size))

    def update_backward(self, grad_output):
        # note that this batch_size is for a single node (may not be the real bs with multi-gpu)
        batch_size = grad_output.size(0)
        self._G.append(grad_output.t() @ (grad_output * batch_size))
