"""
Computes the test accuracy of the model.

author: David-Alexandre Beaupre
date: 2020-05-02
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.metrics as metrics
import utils.misc as misc
from datahandler.LITIV import LITIV
from datahandler.LITIVDataset import TestLITIVDataset
from models.concatnet import ConcatNet
from models.corrnet import CorrNet
from models.stereohrnet import StereoHRNet

from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel

from config import config
from config import update_config

import pprint


def test(model: torch.nn.Module, loader: TestLITIVDataset, name: str, max_disp: int, bsize: int, cuda: bool) \
        -> float:
    """
    test function for either the proposed model, or its individual branches.
    :param model: torch model.
    :param loader: data loader.
    :param name: name of the model.
    :param max_disp: maximum disparity to match patches.
    :param bsize: batch size.
    :param cuda: use GPU or not.
    :return: accuracy value.
    """
    print('testing...')
    model.eval()
    with torch.no_grad():
        correct_1 = 0
        correct_3 = 0
        correct_5 = 0
        for i in range(0, loader.disparity.shape[0], bsize):
            if i == loader.last_batch_idx:
                print(f'\r{i + loader.remainder} / {loader.disparity.shape[0]}', end='', flush=True)
            else:
                print(f'\r{i + bsize} / {loader.disparity.shape[0]}', end='', flush=True)
            rgb, lwir, targets = loader.get_batch()

            disp = torch.arange(start=0, end=(max_disp + 1), dtype=torch.float32)
            disp = disp.repeat(repeats=(bsize, )).reshape(bsize, max_disp + 1)
            weight_corr = torch.zeros(size=(bsize, max_disp + 1), dtype=torch.float32)
            weight_concat = torch.zeros(size=(bsize, max_disp + 1), dtype=torch.float32)
            weight_corr_stage_1 = torch.zeros(size=(bsize, max_disp + 1), dtype=torch.float32)
            weight_concat_stage_1 = torch.zeros(size=(bsize, max_disp + 1), dtype=torch.float32)
            if cuda:
                rgb = rgb.cuda()
                lwir = lwir.cuda()
                targets = targets.cuda()
                disp = disp.cuda()
                weight_corr = weight_corr.cuda()
                weight_concat = weight_concat.cuda()
                weight_corr_stage_1 = weight_corr_stage_1.cuda()
                weight_concat_stage_1 = weight_concat_stage_1.cuda()

            frgb, frgb_stage_1 = model.rgb_features(rgb)
            flwir, flwir_stage_1 = model.lwir_features(lwir)


            for d in range(flwir.shape[3]-flwir.shape[2]): # patch width - individual patch width gives the vumber of disparities in the patch
                lw = flwir[:, :, :, d:d+36]
                lw_stage_1 = flwir_stage_1[:, :, :, d:d+36]

                if name == 'corrnet':
                    correlation = torch.matmul(frgb, lw)
                    correlation = correlation.view(correlation.size(0), -1)
                    corr = torch.softmax(model.correlation_cls(correlation), dim=1)
                    weight_corr[:, d] = corr[:, 1]
                elif name == 'concatnet':
                    concatenation = torch.cat((F.relu(frgb), F.relu(lw)), dim=1)
                    concatenation = concatenation.view(concatenation.size(0), -1)
                    concat = torch.softmax(model.concat_cls(concatenation), dim=1)
                    weight_concat[:, d] = concat[:, 1]
                else:
                    correlation = torch.matmul(frgb, lw)

                    concatenation = torch.cat((F.relu(frgb), F.relu(lw)), dim=1)
                    correlation = correlation.view(correlation.size(0), -1)
                    concatenation = concatenation.view(concatenation.size(0), -1)
                    corr = torch.softmax(model.correlation_cls(correlation), dim=1)
                    concat = torch.softmax(model.concat_cls(concatenation), dim=1)
                    weight_corr[:, d] = corr[:, 1]
                    weight_concat[:, d] = concat[:, 1]


                    correlation_stage_1 = torch.matmul(frgb_stage_1, lw_stage_1)

                    concatenation_stage_1 = torch.cat((F.relu(frgb_stage_1), F.relu(lw_stage_1)), dim=1)
                    correlation_stage_1 = correlation_stage_1.view(correlation_stage_1.size(0), -1)
                    concatenation_stage_1 = concatenation_stage_1.view(concatenation_stage_1.size(0), -1)
                    corr_stage_1 = torch.softmax(model.correlation_cls(correlation_stage_1), dim=1)
                    concat_stage_1 = torch.softmax(model.concat_cls(concatenation_stage_1), dim=1)
                    weight_corr_stage_1[:, d] = corr_stage_1[:, 1]
                    weight_concat_stage_1[:, d] = concat_stage_1[:, 1]

            if name == 'corrnet':
                w_corr = torch.softmax(weight_corr, dim=1)
                w_concat = torch.softmax(weight_corr, dim=1)
            elif name == 'concatnet':
                w_corr = torch.softmax(weight_concat, dim=1)
                w_concat = torch.softmax(weight_concat, dim=1)
            else:
                w_corr = torch.softmax(weight_corr, dim=1)
                w_concat = torch.softmax(weight_concat, dim=1)


                w_corr_stage_1 = torch.softmax(weight_corr_stage_1, dim=1)
                w_concat_stage_1 = torch.softmax(weight_concat_stage_1, dim=1)

            corr_d = torch.sum(w_corr * disp, dim=1)
            concat_d = torch.sum(w_concat * disp, dim=1)
            corr_d_stage_1 = torch.sum(w_corr_stage_1 * disp, dim=1)
            concat_d_stage_1 = torch.sum(w_concat_stage_1 * disp, dim=1)
            dp = (corr_d + concat_d + corr_d_stage_1 + concat_d_stage_1) / 4.0
            correct_1 += metrics.correct_matches_distance_n(dp, targets, 1)
            correct_3 += metrics.correct_matches_distance_n(dp, targets, 3)
            correct_5 += metrics.correct_matches_distance_n(dp, targets, 5)

    accuracy_1 = float(correct_1) / float(loader.disparity.shape[0])
    accuracy_3 = float(correct_3) / float(loader.disparity.shape[0])
    accuracy_5 = float(correct_5) / float(loader.disparity.shape[0])

    return accuracy_1, accuracy_3, accuracy_5


def main() -> None:
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='logs', help='folder to save logs from executions')
    parser.add_argument('--fold', type=int, default=1, help='which fold to test on')
    parser.add_argument('--model', default='stereohrnet', help='name of the model to train')
    parser.add_argument('--datapath', default='/home/beaupreda/litiv/datasets/litiv')
    parser.add_argument('--loadmodel', default='pretrained/stereohrnet/fold1.pt',
                        help='name of the trained model to load, if any')
    parser.add_argument('--max_disparity', type=int, default=64, help='maximum disparity in the dataset')
    parser.add_argument('--patch_size', type=int, default=18, help='half width of the patch')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='enables/disables GPU')  
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    update_config(config, args)

    logger, final_output_dir, _ = create_logger(
    config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    dataset = LITIV(root=args.datapath, psize=args.patch_size, fold=args.fold,)
    dataloader = TestLITIVDataset(dataset.rgb['test'], dataset.lwir['test'], dataset.disp['test'], args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(f'loading model...')
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
        model.cuda()
        criterion.cuda()

    accuracy = test(model, dataloader, args.model, args.max_disparity, args.batch_size, args.cuda)
    print(f'\ntest accuracy n=1: {accuracy[0] * 100:.2f}')
    print(f'\ntest accuracy n=3: {accuracy[1] * 100:.2f}')
    print(f'\ntest accuracy n=5: {accuracy[2] * 100:.2f}')

    print('Fin.')


if __name__ == '__main__':
    main()
