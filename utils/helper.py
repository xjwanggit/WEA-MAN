import shutil
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)  # 因为target存放的是输入类型的标签，故而需要用target.size(0)获得输入的照片的数量
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)  # 在output中按照dim=1取maxk个数，返回的类型中包括选择后的数组,也就是一个包含类别概率的数组以及该数组中各数字所在原位置的索引
    pred = pred.t()  # 进行一个转置
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 这个相当于是torch.eq,其中eq表示的是equal，其实就是对两个张量进行一一对比, expand_as就是讲target输出，扩展成pred从而可进行对比

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)  # 这里view(-1)或者用reshape(-1)其实就是把行列式弄成一串
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, filename='alex_checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}_model_best.pth'.format(filename.split('.')[0]))


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
