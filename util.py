import torch
import shutil


def save_checkpoint(state, is_best, args):
    ''' save the state of training, and if it is
    determined a better module than the current best
    one, save it as the best module, too.
    '''
    torch.save(state, args.resume)
    if is_best:
        shutil.copyfile(args.resume, args.best)


class IOStream():
    ''' Write down the results of each epoch
        into a log file and print it
    '''

    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

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
