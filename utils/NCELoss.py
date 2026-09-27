import torch.nn.functional as F
import torch.nn as nn
import torch

class NCELoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(reduction='mean')):
        super().__init__()
        print('=========using NCE Loss==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = len(prediction)
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)   # label * 10: similar as infoNCE loss with a temperature of 0.1
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss


class DualLoss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(reduction='mean')):
        super().__init__()
        print('=========using DS Loss==========')
        self.error_metric = error_metric

    def forward(self, prediction, label, temp=1000):
        batch_size = len(prediction)
        prediction = prediction * F.softmax(prediction/temp, dim=0) * batch_size
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss

def BYOLLoss(embedding_pred, embedding_target):
    # loss_fn = nn.MSELoss()
    # x=embedding_pred
    # y=embedding_target
    # x = F.normalize(x, dim=-1, p=2)
    # y = F.normalize(y, dim=-1, p=2)

    # aaa=x-y
    # # aaa = F.normalize(aaa, dim=-1, p=2)
    # print('hand', (aaa**2).mean())    # pytorch mse

    # mse = loss_fn(x, y)               # pytorch mse
    # print('mseeeee', mse)
    
    # z = 2 - 2 * (x * y).sum(dim=-1)   # byol mse
    # # zz = x**2 + y**2 - 2* (x * y)   # pytorch mse
    # zz= ((x-y)**2).sum(-1).mean()     # byol mse
    # print('ori loss', zz.mean())

    embedding_pred = embedding_pred / embedding_pred.norm(dim=-1, keepdim=True)
    embedding_target = embedding_target / embedding_target.norm(dim=-1, keepdim=True)
    loss = 2 - 2 * (embedding_pred * embedding_target).sum(dim=-1)
    # print('loss', loss.mean())
    return loss.mean() 