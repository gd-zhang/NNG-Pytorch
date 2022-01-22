import torch
import torch.nn.functional as F
from curvature.curvature import DiagCurvature, KronCurvature
from utils.kfac_utils import try_contiguous


class KronConv2d(KronCurvature):

    def _get_shape(self):
        out_channel, in_channel, ksize0, ksize1 = self._module.weight.data.size()
        in_dim = in_channel * ksize0 * ksize1
        if self.bias:
            return [(in_dim+1, in_dim+1), (out_channel, out_channel)]
        else:
            return [(in_dim, in_dim), (out_channel, out_channel)]

    def _get_grad_mat(self):
        p_grad_mat = self._module.weight.grad.data.view(self._module.weight.grad.data.size(0), -1)
        if self.bias:
            p_grad_mat = torch.cat([p_grad_mat, self._module.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def update_forward(self, data_input):
        conv = self._module
        input_patches = F.unfold(data_input, kernel_size=conv.kernel_size, stride=conv.stride,
                                 padding=conv.padding, dilation=conv.dilation)
        bs, in_dim, spatial_locations = input_patches.size()
        input_patches = input_patches.transpose(1, 2).reshape(-1, in_dim).div(spatial_locations)

        if self.bias is not None:
            input_patches = torch.cat([input_patches, input_patches.new(input_patches.size(0), 1).fill_(1)], 1)
        self._A.append(input_patches.t() @ (input_patches / bs))

    def update_backward(self, grad_output):
        bs, c, h, w = grad_output.shape
        grad_output = grad_output * bs * h * w # note that bs here is only for a single node
        grad_output = grad_output.transpose(1, 2).transpose(2, 3).reshape(-1, c) # (bs * h * w) * c
        self._G.append(grad_output.t() @ (grad_output / grad_output.size(0)))
