import torch
import torch.nn as nn
import torch.nn.functional as F


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x


def _extract_patches(x, kernel_size, stride, padding):
    """
    :param x: The input feature maps.  (batch_size, in_c, h, w)
    :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    :param stride: the stride of conv operation  (tuple of two elements)
    :param padding: number of paddings. be a tuple of two elements
    :return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


def update_running_stats(aa, m_aa, stat_decay):
    # using inplace operation to save memory!
    m_aa *= stat_decay / (1 - stat_decay)
    m_aa += aa
    m_aa *= (1 - stat_decay)


def get_closure(optimizer, model, data, target, criterion=None, approx_type='mc'):
    if criterion is None:
        criterion = nn.CrossEntropyLoss().cuda()

    def turn_off_param_grad():
        for group in optimizer.param_groups:
            for param in group['params']:
                param.requires_grad = False

    def turn_on_param_grad():
        for group in optimizer.param_groups:
            for param in group['params']:
                param.requires_grad = True

    def turn_off_acc_stats():
        for group in optimizer.param_groups:
            if group['curv'] is not None:
                group['curv'].set_acc_stats(False)

    def turn_on_acc_stats():
        for group in optimizer.param_groups:
            if group['curv'] is not None:
                group['curv'].set_acc_stats(True)

    def closure(acc_stats=True):

        if acc_stats:
            turn_on_acc_stats()
            optimizer.zero_grad()
            output = model(data)
            prob = F.softmax(output, dim=1)

            turn_off_param_grad()
            if approx_type == 'mc':
                with torch.no_grad():
                    try:
                        dist = torch.distributions.Categorical(prob)
                        target_ = dist.sample((1, ))[0]
                    except:
                        import pdb
                        pdb.set_trace()
            else:
                target_ = target

            loss = criterion(output, target_)
            loss.backward(retain_graph=True)

            turn_off_acc_stats()
            turn_on_param_grad()
        else:
            optimizer.zero_grad()
            output = model(data)

        loss = criterion(output, target)
        loss.backward()

        return loss, output

    return closure
