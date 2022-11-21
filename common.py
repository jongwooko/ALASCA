import torch
import os
import shutil
import numpy as np

def save_checkpoint(state, epoch, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if epoch % 100 == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_' + str(epoch) + '.pth.tar'))
        
def make_imb_data(max_num, class_num, gamma):
    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    print(class_num_list)
    return list(class_num_list)
        
def load_checkpoint(model, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    model.load_state_dict(torch.load(filepath)['state_dict'])
    return model
        
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    
    if args.dataset == 'cifar10':
        if epoch > 80:
            lr = args.lr * (args.gamma ** 2)
        elif epoch > 40:
            lr = args.lr * args.gamma
        else:
            lr = args.lr
    else:
        if epoch > 120:
            lr = args.lr * (args.gamma ** 2)
        elif epoch > 80:
            lr = args.lr * args.gamma
        else:
            lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr     
    return lr