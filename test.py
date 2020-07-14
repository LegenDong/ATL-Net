from __future__ import print_function

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from PIL import ImageFile

from datasets import ImageFolder
from models import ALTNet
from utils import AverageMeter, print_func, mean_confidence_interval
from utils import accuracy
from utils import prepare_device

mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]


def validate(val_loader, model, criterion, epoch_index, device, fout_file, image2level, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    accuracies = []

    end = time.time()
    with torch.no_grad():
        for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(val_loader):

            way_num = len(support_images)
            shot_num = len(support_images[0])
            query_input = torch.cat(query_images, 0)
            query_targets = torch.cat(query_targets, 0)

            if image2level == 'image2task':
                image_list = []
                for images in support_images:
                    image_list.extend(images)
                support_input = [torch.cat(image_list, 0)]
            else:
                raise RuntimeError

            query_input = query_input.to(device)
            query_targets = query_targets.to(device)
            support_input = [item.to(device) for item in support_input]
            # support_targets = support_targets.to(device)

            # calculate the output
            _, output, _ = model(query_input, support_input)
            output = torch.mean(output.view(-1, way_num, shot_num), dim=2)
            loss = criterion(output, query_targets)

            # measure accuracy and record loss
            prec1, _ = accuracy(output, query_targets, topk=(1, 3))
            losses.update(loss.item(), query_input.size(0))
            top1.update(prec1[0], query_input.size(0))
            accuracies.append(prec1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print the intermediate results
            if episode_index % print_freq == 0 and episode_index != 0:
                info_str = ('Test-({0}): [{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                            .format(epoch_index, episode_index, len(val_loader), batch_time=batch_time,
                                    loss=losses, top1=top1))
                print_func(info_str, fout_file)
    return top1.avg, accuracies


def main(result_path, epoch_num):
    config = json.load(open(os.path.join(result_path, 'config.json')))

    fout_path = os.path.join(result_path, 'test_info.txt')
    fout_file = open(fout_path, 'a+')
    print_func(config, fout_file)

    trsfms = transforms.Compose([
        transforms.Resize((config['general']['image_size'], config['general']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    model = ALTNet(**config['arch'])
    print_func(model, fout_file)

    state_dict = torch.load(os.path.join(result_path, '{}_best_model.pth'.format(config['data_name'])))
    model.load_state_dict(state_dict)

    if config['train']['loss']['name'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(**config['train']['loss']['args'])
    else:
        raise RuntimeError

    device, _ = prepare_device(config['n_gpu'])
    model = model.to(device)
    criterion = criterion.to(device)

    total_accuracy = 0.0
    total_h = np.zeros(epoch_num)
    total_accuracy_vector = []
    for epoch_idx in range(epoch_num):
        test_dataset = ImageFolder(
            data_root=config['general']['data_root'], mode='test', episode_num=600,
            way_num=config['general']['way_num'], shot_num=config['general']['shot_num'],
            query_num=config['general']['query_num'], transform=trsfms,
        )

        print_func('The num of the test_dataset: {}'.format(len(test_dataset)), fout_file)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config['test']['batch_size'], shuffle=True,
            num_workers=config['general']['workers_num'], drop_last=True, pin_memory=True
        )

        print_func('============ Testing on the test set ============', fout_file)
        _, accuracies = validate(test_loader, model, criterion, epoch_idx, device, fout_file,
                                 config['general']['image2level'],
                                 config['general']['print_freq'])
        test_accuracy, h = mean_confidence_interval(accuracies)
        print_func("Test Accuracy: {}\t h: {}".format(test_accuracy, h[0]), fout_file)

        total_accuracy += test_accuracy
        total_accuracy_vector.extend(accuracies)
        total_h[epoch_idx] = h

    aver_accuracy, _ = mean_confidence_interval(total_accuracy_vector)
    print_func('Aver Accuracy: {:.3f}\t Aver h: {:.3f}'.format(aver_accuracy, total_h.mean()), fout_file)
    print_func('............Testing is end............', fout_file)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    parser = argparse.ArgumentParser(description='ATL_Net in PyTorch')
    parser.add_argument('-r', '--result_path', default=None, type=str, help='')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('-e', '--epoch_num', default=5, type=int, help='')

    args = parser.parse_args()

    if not args.result_path:
        raise AssertionError("result path need to be specified")
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(args.result_path, args.epoch_num)
