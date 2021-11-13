import torch
from torch.optim.optimizer import Optimizer, required
import torch.distributed as dist
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from mpl_toolkits.axisartist.axislines import SubplotZero
import random
import time


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

    ###===================================================Baseline=====================================================###
    def step_base(self, closure=None):
        """
        A single optimization step without sparsification.
        """
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
                # average the gradients of all workers
                dist.all_reduce(d_p)  
                d_p /= float(dist.get_world_size())
                p.data.add_(-group['lr'], d_p)

        up_ratio = 1.

        return loss, up_ratio


    ###===================================================GRSGD=====================================================###
    def step_grsgd(self, closure=None):
        """
        A single optimization step using Global Renovating SGD.
        """
        loss = None
        if closure is not None:
            loss = closure()

        up_number = 0
        total_number = 0

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # get the previous-round gradient
                if 'g_global_v' not in self.state[p]:
                    self.state[p]['g_global_v'] = torch.zeros_like(d_p).detach()
                g_global_v = self.state[p]['g_global_v']
                midst = g_global_v.mul(d_p)
                # renovate the sparsified same-sign part with the previous-round gradient 
                d_p_cut = torch.where(midst <= 0, d_p, g_global_v)

                # calculate the upload ratio
                one_zero = torch.where(midst <= 0, torch.tensor(1.).cuda(), torch.tensor(0.).cuda())
                total_number += d_p.numel()
                up_number += one_zero.sum().item()

                # average the gradients of all workers
                dist.all_reduce(d_p_cut)
                d_p_cut /= float(dist.get_world_size())
                # update the previous-round gradient
                self.state[p]['g_global_v'] = torch.clone(d_p_cut).detach()
                p.data.add_(-group['lr'], d_p_cut)

        up_ratio = float(up_number) / float(total_number)

        return loss, up_ratio


    ###===================================================Topk=====================================================###
    def step_topk(self, closure=None):
        """
        A single optimization step using Top-k.
        """
        upload_ratio = 0.45
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
                # get the top-k values and indices
                k = max(1, int(d_p.numel() * upload_ratio))
                d_p_line = d_p.flatten().cuda()
                _, indices = torch.topk(d_p_line.abs(), k)
                values = d_p_line[indices].cuda()
                # Recover the sparsified gradient
                d_p_recover = torch.zeros(d_p_line.numel(), dtype=values.dtype, layout=values.layout,
                                          device=values.device)
                d_p_recover.scatter_(0, indices, values)
                d_p_cut = d_p_recover.view(d_p.size()).cuda()

                # average the gradients of all workers
                dist.all_reduce(d_p_cut)
                d_p_cut /= float(dist.get_world_size())
                p.data.add_(-group['lr'], d_p_cut)

        up_ratio = upload_ratio

        return loss, up_ratio


    ###===================================================mTopk=====================================================###
    def step_mtopk(self, closure=None):
        """
        A single optimization step using mTop-k (Top-k with memory).
        """
        upload_ratio = 0.45
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                param_state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # get the residual and compensate
                if 'residual' not in param_state:
                    param_state['residual'] = torch.zeros_like(d_p).detach()
                d_p = d_p + param_state['residual']
                # get the top-k values and indices
                k = max(1, int(d_p.numel() * upload_ratio))
                d_p_line = d_p.flatten().cuda()
                _, indices = torch.topk(d_p_line.abs(), k)
                values = d_p_line[indices].cuda()
                # Recover the sparsified gradient
                d_p_recover = torch.zeros(d_p_line.numel(), dtype=values.dtype, layout=values.layout,
                                          device=values.device)
                d_p_recover.scatter_(0, indices, values)
                d_p_cut = d_p_recover.view(d_p.size()).cuda()

                # update the residual
                residual = d_p - d_p_cut
                param_state['residual'] = torch.clone(residual).detach()
                # average the gradients of all workers
                dist.all_reduce(d_p_cut)
                d_p_cut /= float(dist.get_world_size())
                p.data.add_(-group['lr'], d_p_cut)

        up_ratio = upload_ratio

        return loss, up_ratio


    ###===================================================TCS=====================================================###
    def step_tcs(self, closure=None):
        """
        A single optimization step using TCS.
        """
        global_ratio = 0.4
        local_ratio = 0.05
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                param_state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        # param_state['momentum_buffer'] = torch.zeros_like(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
                commu_term = d_p
                # get the residual and compensate
                if 'residual' not in param_state:
                    param_state['residual'] = torch.zeros_like(d_p).detach()
                    param_state['global_indices'] = torch.tensor([i for i in range(d_p.flatten().numel())]).cuda()
                commu_term = commu_term + param_state['residual']
                commu_term_line = commu_term.flatten()
                # retain the values corresponding to the global top-k indices
                global_indices = param_state['global_indices']
                l_global_values = commu_term_line[global_indices].cuda()
                commu_term_recover = torch.zeros(commu_term_line.numel(), dtype=l_global_values.dtype, layout=l_global_values.layout,
                                          device=l_global_values.device)
                commu_term_recover.scatter_(0, global_indices, l_global_values)
                # retain the values corresponding to the local top-k indices
                local_term_line = commu_term_line - commu_term_recover
                local_k = max(1, int(local_term_line.numel() * local_ratio))
                _, local_indices = torch.topk(local_term_line.abs(), local_k)
                local_values = local_term_line[local_indices].cuda()
                commu_term_recover.scatter_(0, local_indices, local_values)
                commu_term_cut = commu_term_recover.view(commu_term.size()).cuda()

                # update the residual
                param_state['residual'] = torch.clone(commu_term-commu_term_cut).detach()
                # average the gradients of all workers
                dist.all_reduce(commu_term_cut)
                commu_term_cut /= float(dist.get_world_size())
                p.data.add_(-group['lr'], commu_term_cut)

                # update the global top-k indices
                global_term = torch.clone(commu_term_cut).detach()
                global_k = max(1, int(global_term.numel() * global_ratio))
                global_term_line = global_term.flatten().cuda()
                _, new_global_indices = torch.topk(global_term_line.abs(), global_k)
                param_state['global_indices'] = new_global_indices

        up_ratio = global_ratio+local_ratio

        return loss, up_ratio


    ###====================================================DGC======================================================###
    def step_dgc(self, closure=None):
        """
        A single optimization step using DGC.
        """
        upload_ratio = 0.45
        up_number = 0
        total_number = 0

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                param_state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = torch.clone(buf).detach()

                if 'residual' not in param_state:
                    param_state['residual'] = torch.zeros_like(d_p).detach()
                # local gradient clipping
                tensor_squ_sum = torch.sum(d_p * d_p).cuda()
                dist.all_reduce(tensor_squ_sum)
                clipping_val = torch.sqrt(tensor_squ_sum / dist.get_world_size())
                d_p = d_p.clamp(-clipping_val, clipping_val)
                # compensate
                d_p = d_p + param_state['residual']
                # top-k sparsification with 1% sampling
                d_p_line = d_p.flatten()
                sample_shape = [max(1, int(d_p_line.numel() * 0.01))]
                sample_index = torch.empty(sample_shape).uniform_(0, d_p_line.numel()).type(torch.long).cuda()
                sample_tensor = torch.index_select(d_p_line, dim=0, index=sample_index)
                k = max(1, int(d_p.numel() * upload_ratio * 0.01))
                sample_abs = sample_tensor.abs()
                d_p_abs = d_p.abs()
                values, indices = torch.topk(sample_abs, k)
                threshold = values.min()
                d_p_cut = torch.where(d_p_abs >= threshold, d_p, torch.tensor(0.).cuda())

                up_number += d_p_cut.bool().sum()
                total_number += d_p.numel()
                # update the residual
                residual = d_p - d_p_cut
                param_state['residual'] = torch.clone(residual).detach()
                # average the gradients of all workers
                dist.all_reduce(d_p_cut)
                d_p_cut /= float(dist.get_world_size())
                p.data.add_(-group['lr'], d_p_cut)

        up_ratio = float(up_number) / float(total_number)

        return loss, up_ratio

