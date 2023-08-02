import torch
from torch import nn


sigmoid = nn.Sigmoid()

# Standard loss
def loss_pointwise_negative_Standard(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):
    loss_y = (((y_deepfm_pos - y) ** 2)).sum()
    loss_y_neg = (((y_deepfm_neg - 0) ** 2)).sum()

    loss = loss_y + loss_y_neg
    return loss

def loss_pointwise_Standard(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):
    loss_y = (((y_deepfm_pos - y) ** 2)).sum()

    loss = loss_y
    return loss

def loss_pairwise_Standard(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):
    bpr_click = - (sigmoid(y_deepfm_pos - y_deepfm_neg).log()).sum()

    loss = bpr_click
    return loss

def loss_pairwise_pointwise_Standard(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):
    loss_y = (((y_deepfm_pos - y) ** 2)).sum()
    bpr_click = - (sigmoid(y_deepfm_pos - y_deepfm_neg).log()).sum()

    loss = loss_y + args.bpr_weight * bpr_click
    return loss

# IPS loss
def loss_pointwise_negative_IPS(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):
    loss_y = (((y_deepfm_pos - y) ** 2) * score).sum()
    loss_y_neg = (((y_deepfm_neg - 0) ** 2)).sum()

    loss = loss_y + loss_y_neg
    return loss

def loss_pointwise_IPS(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):
    loss_y = (((y_deepfm_pos - y) ** 2) * score).sum()

    loss = loss_y
    return loss


def loss_pairwise_IPS(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):
    bpr_click = - (sigmoid(y_deepfm_pos - y_deepfm_neg).log()*score).sum()

    loss = bpr_click
    return loss

def loss_pairwise_pointwise_IPS(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):
    loss_y = (((y_deepfm_pos - y) ** 2) * score).sum()
    bpr_click = - (sigmoid(y_deepfm_pos - y_deepfm_neg).log() * score).sum()

    loss = loss_y + args.bpr_weight * bpr_click
    return loss

# Advance loss
def process_logit(y_deepfm_pos, score, alpha_u=None, beta_i=None, args=None):
    if alpha_u is not None:
        score_new = score * alpha_u * beta_i
        loss_ab = ((alpha_u - 1) ** 2).mean() + ((beta_i - 1) ** 2).mean()
    else:
        score_new = score
        loss_ab = 0
    loss_ab = args.lambda_ab * loss_ab
    y_weighted = 1 / (1 + score_new) * y_deepfm_pos
    return y_weighted, loss_ab


def loss_pointwise_negative(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None,
                            log_var_neg=None):
    y_weighted, loss_ab = process_logit(y_deepfm_pos, score, alpha_u=alpha_u, beta_i=beta_i, args=args)

    if log_var is not None:
        inv_var = torch.exp(-log_var)
        inv_var_neg = torch.exp(-log_var_neg)
        loss_var_pos = log_var.sum()
        loss_var_neg = log_var_neg.sum()
    else:
        inv_var = 1
        inv_var_neg = 1
        loss_var_pos = 0
        loss_var_neg = 0

    loss_y = (((y_weighted - y) ** 2) * inv_var).sum()
    loss_y_neg = (((y_deepfm_neg - 0) ** 2) * inv_var_neg).sum()

    loss = loss_y + loss_y_neg + loss_ab + loss_var_pos + loss_var_neg
    return loss


def loss_pointwise(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None,
                   log_var_neg=None):
    y_weighted, loss_ab = process_logit(y_deepfm_pos, score, alpha_u=alpha_u, beta_i=beta_i, args=args)

    if log_var is not None:
        inv_var = torch.exp(-log_var)
        loss_var_pos = log_var.sum()
    else:
        inv_var = 1
        loss_var_pos = 0

    loss_y = (((y_weighted - y) ** 2) * inv_var).sum()

    loss = loss_y + loss_ab + loss_var_pos
    return loss


def loss_pairwise(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None,
                  log_var_neg=None):
    y_weighted, loss_ab = process_logit(y_deepfm_pos, score, alpha_u=alpha_u, beta_i=beta_i, args=args)
    # loss_y = ((y_exposure - y) ** 2).sum()

    if log_var is not None:
        inv_var = torch.exp(-log_var)
        inv_var_neg = torch.exp(-log_var_neg)
        loss_var_pos = log_var.sum()
        loss_var_neg = log_var_neg.sum()
    else:
        inv_var = 1
        inv_var_neg = 1
        loss_var_pos = 0
        loss_var_neg = 0

    bpr_click = - (sigmoid(y_weighted - y_deepfm_neg).log() * inv_var * inv_var_neg).sum()
    
    loss = bpr_click + loss_ab + loss_var_pos + loss_var_neg
    return loss


def loss_pairwise_pointwise(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None,
                            log_var_neg=None):
    y_weighted, loss_ab = process_logit(y_deepfm_pos, score, alpha_u=alpha_u, beta_i=beta_i, args=args)
    if log_var is not None:
        inv_var = torch.exp(-log_var)
        loss_var_pos = log_var.sum()
    else:
        inv_var = 1
        loss_var_pos = 0
    loss_y = (((y_weighted - y) ** 2) * inv_var).sum()
    bpr_click = - sigmoid(y_weighted - y_deepfm_neg).log().sum()
    
    loss = loss_y + args.bpr_weight * bpr_click + loss_ab + loss_var_pos
    return loss