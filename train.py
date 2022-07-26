"""
training script for the proposed model.

author: David-Alexandre Beaupre
date: 2020-04-29
"""

import os
import argparse
import time as t
import datetime as dt

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

import utils.misc as misc
from datahandler.LITIV import LITIV
from datahandler.LITIVDataset import TrainLITIVDataset
from models.concatnet import ConcatNet
from models.corrnet import CorrNet
from models.stereohrnet import StereoHRNet
from utils.graphs import Graph
from utils.logs import Logs

from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel

from config import config
from config import update_config


import pprint


def training(model: torch.nn.Module, loader: TrainLITIVDataset, criterion: torch.nn.Module, optimizer: torch.optim,
             iterations: int, name: str, cuda: bool, bsize: int) -> (float, float):
    """
    training function to train either the proposed model, or its individual branches.
    :param model: torch model.
    :param loader: data loader.
    :param criterion: torch loss function.
    :param optimizer: torch optimizer.
    :param iterations: number of training iterations.
    :param name: name of the model.
    :param cuda: use GPU or not.
    :param bsize: batch size.
    :return: accuracy and loss values.
    """
    print('training...')
    model.train()
    correct_corr = 0
    correct_concat = 0
    correct_corr_stage1 = 0
    correct_concat_stage1 = 0
    total_loss = 0.0
    for i in range(0, iterations):
        print(f'\r{i + 1} / {iterations}', end='', flush=True)
        rgb, lwir, targets = loader.get_batch()

        if cuda:
            rgb = rgb.cuda()
            lwir = lwir.cuda()
            targets = targets.cuda()
        if name == 'corrnet':
            corr = model(rgb, lwir)
            loss = criterion(corr, targets)
            _, predictions_corr = torch.max(corr, dim=1)
            correct_corr += torch.sum(predictions_corr == targets)
        elif name == 'concatnet':
            concat = model(rgb, lwir)
            loss = criterion(concat, targets)
            _, predictions_concat = torch.max(concat, dim=1)
            correct_concat += torch.sum(predictions_concat == targets)
        else:
            corr, concat, corr_stage1, concat_stage1 = model(rgb, lwir)
            loss = criterion(corr, targets) + criterion(concat, targets) + criterion(corr_stage1, targets) + criterion(concat_stage1, targets)
            _, predictions_corr = torch.max(corr, dim=1)
            _, predictions_concat = torch.max(concat, dim=1)

            _, predictions_corr_stage1 = torch.max(corr_stage1, dim=1)
            _, predictions_concat_stage1 = torch.max(concat_stage1, dim=1)

            correct_corr += torch.sum(predictions_corr == targets)
            correct_concat += torch.sum(predictions_concat == targets)
            correct_corr_stage1 += torch.sum(predictions_corr_stage1 == targets)
            correct_concat_stage1 += torch.sum(predictions_concat_stage1 == targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss

    accuracy_corr = float(correct_corr) / float(iterations * bsize)
    accuracy_concat = float(correct_concat) / float(iterations * bsize)
    accuracy_corr_stage1 = float(correct_corr_stage1) / float(iterations * bsize)
    accuracy_concat_stage1 = float(correct_concat_stage1) / float(iterations * bsize)
    total_loss = float(total_loss) / float(iterations * bsize)

    return max(accuracy_corr, accuracy_concat, accuracy_corr_stage1, accuracy_concat_stage1), total_loss



def validation(model: torch.nn.Module, loader: TrainLITIVDataset, criterion: torch.nn.Module, iterations: int,
               name: str, cuda: bool, bsize: int) -> (float, float):
    """
    validation function to test either the proposed model, or its individual branches.
    :param model: torch model.
    :param loader: data loader.
    :param criterion: torch loss function.
    :param iterations: number of validation iterations.
    :param name: name of the model.
    :param cuda: use GPU or not.
    :param bsize: batch size.
    :return: accuracy and loss values.
    """
    print('validation...')
    model.eval()
    with torch.no_grad():
        correct_corr = 0
        correct_concat = 0
        correct_corr_stage1 = 0
        correct_concat_stage1 = 0
        total_loss = 0.0
        for i in range(0, iterations):
            print(f'\r{i + 1} / {iterations}', end='', flush=True)
            rgb, lwir, targets = loader.get_batch()
            if cuda:
                rgb = rgb.cuda()
                lwir = lwir.cuda()
                targets = targets.cuda()
            if name == 'corrnet':
                corr = model(rgb, lwir)
                loss = criterion(corr, targets)
                _, predictions_corr = torch.max(corr, dim=1)
                correct_corr += torch.sum(predictions_corr == targets)
            elif name == 'concatnet':
                concat = model(rgb, lwir)
                loss = criterion(concat, targets)
                _, predictions_concat = torch.max(concat, dim=1)
                correct_concat += torch.sum(predictions_concat == targets)
            else:
                corr, concat, corr_stage1, concat_stage1 = model(rgb, lwir)
                loss = criterion(corr, targets) + criterion(concat, targets) + criterion(corr_stage1, targets) + criterion(concat_stage1, targets)
                _, predictions_corr = torch.max(corr, dim=1)
                _, predictions_concat = torch.max(concat, dim=1)

                _, predictions_corr_stage1 = torch.max(corr_stage1, dim=1)
                _, predictions_concat_stage1 = torch.max(concat_stage1, dim=1)

                correct_corr += torch.sum(predictions_corr == targets)
                correct_concat += torch.sum(predictions_concat == targets)
                correct_corr_stage1 += torch.sum(predictions_corr_stage1 == targets)
                correct_concat_stage1 += torch.sum(predictions_concat_stage1 == targets)
            total_loss += loss

    accuracy_corr = float(correct_corr) / float(iterations * bsize)
    accuracy_concat = float(correct_concat) / float(iterations * bsize)
    accuracy_corr_stage1 = float(correct_corr_stage1) / float(iterations * bsize)
    accuracy_concat_stage1 = float(correct_concat_stage1) / float(iterations * bsize)
    total_loss = float(total_loss) / float(iterations * bsize)

    return max(accuracy_corr, accuracy_concat, accuracy_corr_stage1, accuracy_concat_stage1), total_loss

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='logs', help='folder to save logs from executions')
    parser.add_argument('--fold', type=int, default=1, help='which fold to test on')
    parser.add_argument('--model', default='concatcorr', help='name of the model to train (filename without the .py)')
    parser.add_argument('--datapath', default='/home/beaupreda/litiv/datasets/litiv')
    parser.add_argument('--loadmodel', default=None, help='name of the trained model to load, if any')
    parser.add_argument('--patch_size', type=int, default=18, help='half width of the patch')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--train_iterations', type=int, default=200, help='number of training iterations per epochs')
    parser.add_argument('--val_iterations', type=int, default=100, help='number of validation iterations per epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='enables/disables GPU')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    return parser.parse_args()

def main() -> None:
    torch.manual_seed(42)


    args = arg_parser()
    update_config(config, args)

    if not os.path.isdir(args.save):
        os.mkdir(args.save)

    now = dt.datetime.now()
    savepath = os.path.join(args.save, now.strftime('%Y%m%d-%H%M%S'))
    save_logger = Logs(savepath)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print('Cuda Status:', torch.cuda.is_available())

    print('loading dataset...')
    dataset = LITIV(root=args.datapath, psize=args.patch_size, fold=args.fold)
    train_loader = TrainLITIVDataset(dataset.rgb['train'], dataset.lwir['train'], dataset.disp['train'], 'train', args)
    validation_loader = TrainLITIVDataset(dataset.rgb['validation'], dataset.lwir['validation'],
                                          dataset.disp['validation'], 'validation', args)


    # if args.seed > 0:
    #     import random
    #     print('Seeding with', args.seed)
    #     random.seed(args.seed)
    #     torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    # cudnn.benchmark = config.CUDNN.BENCHMARK
    # cudnn.deterministic = config.CUDNN.DETERMINISTIC
    # cudnn.enabled = config.CUDNN.ENABLED


    print('loading model...')
    if args.model == 'corrnet':
        model = CorrNet(config)
    elif args.model == 'concatnet':
        model = ConcatNet(config)
    else:
        model = StereoHRNet(config)

    if args.loadmodel is not None:
        parameters = torch.load(args.loadmodel)
        model.load_state_dict(parameters['state_dict'])
    print(f'number of parameters = {misc.get_number_parameters(model)}\n')

    criterion = nn.CrossEntropyLoss(reduction='sum')

    if args.cuda:
        # gpus = list(config.GPUS)
        # print("CUDA activated...")
        # print("GPUs: {0}".format(gpus))
        # print()
        # model = nn.DataParallel(model, device_ids=gpus).cuda()
        model.cuda()
        criterion.cuda()

    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=0.0005)

    train_losses = []
    validation_losses = []
    train_accuracies = [0.0]
    validation_accuracies = [0.0]
    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        dawn = t.time()
        print(f'******** EPOCH {epoch} ********')
        if (epoch - 1) % 40 == 0 and (epoch - 1) != 0:
            misc.adjust_learning_rate(optimizer, args)


        train_accuracy, train_loss = training(model, train_loader, criterion, optimizer, args.train_iterations,
                                              args.model, args.cuda, args.batch_size)
        print(f'\ntrain loss = {train_loss:.2f}, train accuracy = {train_accuracy * 100:.2f}')
        validation_accuracy, validation_loss = validation(model, validation_loader, criterion, args.val_iterations,
                                                          args.model, args.cuda, args.batch_size)
        print(f'\nvalidation loss = {validation_loss:.2f}, validation accuracy = {validation_accuracy * 100:.2f}')
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        dusk = t.time()
        print(f'elapsed time: {dusk - dawn:.2f} s')

        save_logger.save_accuracy_loss(epoch, train_accuracy, validation_accuracy, train_loss, validation_loss)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            misc.save_model(savepath, model, epoch, train_loss, validation_loss)
        elif epoch % 10 == 0:
            misc.save_model(savepath, model, epoch, train_loss, validation_loss)
        print(f'******** END EPOCH {epoch} ********\n')

    fig_accuracy, ax_accuracy = plt.subplots()
    fig_loss, ax_loss = plt.subplots()

    accuracy_graph = Graph(savepath, fig_accuracy, ax_accuracy, loss=False)
    loss_graph = Graph(savepath, fig_loss, ax_loss, loss=True)

    accuracy_graph.create(train_accuracies, validation_accuracies)
    loss_graph.create(train_losses, validation_losses)

    accuracy_graph.save()
    loss_graph.save()

    print("Best accuracy:", best_accuracy)
    print("Best epoch:", best_epoch)

    print('Fin.')


if __name__ == '__main__':
    main()
