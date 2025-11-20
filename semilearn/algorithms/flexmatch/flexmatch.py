
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from .utils import FlexMatchThresholdingHook

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('flexmatch')
class FlexMatch(AlgorithmBase):
    """
        FlexMatch algorithm (https://arxiv.org/abs/2110.08263).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ulb_dest_len (`int`):
                Length of unlabeled data
            - thresh_warmup (`bool`, *optional*, default to `True`):
                If True, warmup the confidence threshold, so that at the beginning of the training, all estimated
                learning effects gradually rise from 0 until the number of unused unlabeled data is no longer
                predominant

        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # flexmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, thresh_warmup=args.thresh_warmup)
        self.p_model = torch.ones((self.num_classes)) / self.num_classes
        self.margin = 6
        self.lambda_m = 0.1
        self.lambda_n = 1
        self.K = 8
        self.global_topk_conf = None
    
    def init(self, T, p_cutoff, hard_label=True, thresh_warmup=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.thresh_warmup = thresh_warmup

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FlexMatchThresholdingHook(ulb_dest_len=self.args.ulb_dest_len, num_classes=self.num_classes, thresh_warmup=self.args.thresh_warmup), "MaskingHook")
        super().set_hooks()

    def calculate_k_i(self, prob_w, mask, m, num_classes, batch_size):                  
        k_list = []
        prob_s_multipy = torch.ones_like(prob_w)
        sorted_probs, sorted_idx = torch.sort(prob_w, dim=-1, descending=True)
        topk_cumsum = torch.cumsum(sorted_probs, dim=-1)        
        
        if  self.global_topk_conf == None:
            self.global_topk_conf = torch.zeros(num_classes)
            for t in range(num_classes):
                self.global_topk_conf[t] = (t+1) / num_classes

        self.global_topk_conf = self.global_topk_conf * m + (1-m) * torch.mean(topk_cumsum, dim=0).cpu()
        for b in range(batch_size):
            for t in range(num_classes):
                if (topk_cumsum[b][t] >= self.global_topk_conf[t]) or (t+1 >= self.K):                    
                    prob_s_multipy[b, sorted_idx[b, range(t + 1)]] = 0                    
                    break       
        return prob_s_multipy
        
    def calibration_loss(self, inputs, mask, p_model):                
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        mod =torch.max(p_model, dim=-1)[0]/p_model
        loss_margin = (F.relu(diff - self.margin*mod)*(mask.unsqueeze(-1))).mean()            
        
        return loss_margin

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False, idx_ulb=idx_ulb)
            
            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)
            if not self.p_model.is_cuda:                                      
                self.p_model = self.p_model.to(prob_w.device)
            self.p_model = self.p_model * 0.999 + (1 - 0.999) * prob_w.mean(dim=0)
            # NCP_loss
            ncp_loss = torch.mean((1-mask)*torch.mean(prob_s*self.calculate_k_i(prob_w, mask = mask, m=0.999, num_classes=prob_w.shape[1], batch_size=prob_w.shape[0]), dim=-1))
            # CAM_loss
            calibration_loss = self.calibration_loss(logits_x_ulb_s, mask, self.p_model)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_m * calibration_loss + self.lambda_n * ncp_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict
        

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['classwise_acc'] = self.hooks_dict['MaskingHook'].classwise_acc.cpu()
        save_dict['selected_label'] = self.hooks_dict['MaskingHook'].selected_label.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].classwise_acc = checkpoint['classwise_acc'].cuda(self.gpu)
        self.hooks_dict['MaskingHook'].selected_label = checkpoint['selected_label'].cuda(self.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--thresh_warmup', str2bool, True),
        ]
