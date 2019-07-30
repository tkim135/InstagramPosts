import resnet_data_utils
import time
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import models
import os
import glob
from util import get_args

class Net(nn.Module):

    def __init__(self):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(Net, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.avgpool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(2048, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        features = self.resnet(x)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)  # reshape
        #return features
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        features = self.fc3(features)
        features = torch.sigmoid(features)
        return features

#Combined(Resnet50(), efef)


# printing out for a few sample images
def visualize_output():
    if not os.path.isdir('visuals/'):
        os.makedirs('visuals/')
    state_dict = load_state_dict()
    network = state_dict['network']
    _, val_loader = resnet_data_utils.get_dataloaders(batch_sz=1) # want one image
    for i, (input, label) in enumerate(val_loader):
        outputs = network(input)
        label = torch.argmax(outputs, dim=1).item()
        label = 'AD' if label == 0 else 'NOT_AD'
        print('Saving under visuals/' + str(np.random.randint(10000)) + '_class=' + label + '.png')
        save_image(input, 'visuals/' + str(np.random.randint(10000)) + '_class=' + label + '.png')
        if i == 10: break


def load_state_dict():
    '''
    Always loads the latest saved network state.
    '''
    all_weight_files = glob.glob('weights/*')
    all_weight_files = sorted(all_weight_files)
    print('Loading ' + all_weight_files[-1] + '...')
    state_dict = torch.load(all_weight_files[-1])
    print('Finished loading!')
    return state_dict


def validate(val_loader, model, criterion, args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1,
                             prefix='Test: ')

    print('\n --- BEGIN VALIDATION PASS --- ')

    # switch to evaluate mode
    model.eval()

    guess = np.array([])
    actual = np.array([])
    
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            # No GPU at the moment
            # if args.gpu is not None:
            #     input = input.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc_tuple = accuracy(output, target)
            acc1 = acc_tuple[0][0]
            #import pdb; pdb.set_trace()
            for val in acc_tuple[1][0]:
                assert(val==0 or val==1)
            for val in acc_tuple[2][0]:
                assert(val==0 or val==1)
            guess = np.concatenate((guess, acc_tuple[1][0]))
            actual = np.concatenate((actual, acc_tuple[2][0]))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))

        print(' --- END VALIDATION PASS --- \n')
    print(actual)
    print(guess)
    print(metrics.classification_report(actual, guess, digits=3))
    return top1.avg


# Already saving during training every 100 mini-batches
# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        _, targets = target.topk(maxk, 1, True, True)
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, pred.numpy(), targets.t().numpy()


def main(val=False):
    # Sanity check
    network = Net()
    # Replace with image size (channel, width, height)
    random_input = torch.zeros((1, 3, 224, 224))
    outputs = network(random_input)
    print(outputs.size())

    # Ryan will write this
    train_loader, val_loader = resnet_data_utils.get_dataloaders()

    if val:
        state_dict = load_state_dict()
        validate(val_loader, state_dict['network'], state_dict['criterion'])
    else:
        train(network, train_loader, val_loader)


def train(network, train_loader, val_loader):
    # THIS IS BAD. PLS REFACTOR
    learning_rate = 1e-2
    momentum = 0.9
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.BCELoss()
    num_epochs = 10

    for epoch in range(num_epochs):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                                prefix="Epoch: [{}]".format(epoch))

        running_loss = 0.0
        end = time.time()
        for i, data in enumerate(train_loader):
            input, target = data
            # measure data loading time
            data_time.update(time.time() - end)

            '''# No GPU yet
            args = get_args()
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)'''
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            input.to(device)
            target.to(device)

            # compute output
            output = network(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)[0][0]
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                progress.print(i)

            # Save weights
            if i % 100 == 99:
                print('outputs:', output)
                # validate(val_loader, network, criterion)
                if not os.path.isdir('weights/'):
                    os.makedirs('weights/')
                torch.save({
                        'network': network,
                        'optimizer': optimizer,
                        'criterion': criterion,
                    },
                    'weights/weights_epoch_' + str(epoch) + '_iteration_' + str(i).zfill(6) + '.pt')


    # Training loop

if __name__ == '__main__':
     #visualize_output()
     main(val=False)
