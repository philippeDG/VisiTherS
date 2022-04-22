"""
information about the LITIV 2014 dataset.

author: David-Alexandre Beaupre
date: 2020-04-27
"""

import os
from collections import defaultdict
from typing import DefaultDict, List

import utils.misc as misc
from datahandler.dataset import Dataset
from utils.enums import Datasets


class LITIV2014(Dataset):
    def __init__(self, root: str = None, psize: int = None, height: int = 360, width: int = 480, fold: int = None):
        """
        represents all information unique to LITIV 2014 dataset.
        :param root: path to the folder containing LITIV 2014 folder.
        :param psize: half size of the patch.
        :param height: image height.
        :param width: image width.
        :param fold: number identifying which fold to keep as testing data.
        """
        super(LITIV2014, self).__init__(root, psize, height, width, Datasets.LITIV2014, fold)

        rgb, lwir, mrgb, mlwir, disparity, drange = self._prepare()
        mirrored = self._mirror(rgb, lwir, mrgb, mlwir, disparity, drange)
        self._add_points(rgb, lwir, mrgb, mlwir, disparity, drange, mirrored)

    def _prepare(self) -> (DefaultDict[str, List[str]], DefaultDict[str, List[str]], DefaultDict[str, List[str]],
                           DefaultDict[str, List[str]], DefaultDict[str, List[str]], DefaultDict[str, List[str]]):
        """
        aggregates all images and disparity files from the original LITIV 2014 folder.
        :return: void.
        """
        print(f'preparing LITIV2014 dataset...')
        rgb_paths = defaultdict(list)
        lwir_paths = defaultdict(list)
        mask_rgb_paths = defaultdict(list)
        mask_lwir_paths = defaultdict(list)
        disparity_paths = defaultdict(list)
        drange_paths = defaultdict(list)

        root = os.path.join(self.root, 'BilodeauetAlInfraredDataset/Dataset')

        im_root = root
        #im_root = os.path.join( "/store/travail/philippeDG/preprocessing/4d", 'BilodeauetAlInfraredDataset/Dataset')


        videos = ['vid1', 'vid2', 'vid3']
        cuts = ['cut1', 'cut2']
        vid1_scenes = ['1Person', '2Person', '3Person', '4Person']
        vid2_cut1_scenes = ['1Person', '2Person']
        vid2_cut2_scenes = ['2Person', '3Person', '4Person']
        vid3_scenes = ['1Person', '2Person', '3Person', '4Person', '5Person']
        vid1_invalid_rgb = ['Vis3586.jpg', 'Vis3746.jpg','Vis3586.png', 'Vis3746.png']
        vid1_invalid_lwir = ['IR3586.jpg', 'IR3748.jpg', 'IR3586.png', 'IR3748.png']
        vid2_invalid_rgb =  ['Vis626.jpg', 'Vis638.jpg', 'Vis626.png', 'Vis638.png']
        vid2_invalid_lwir = ['IR626.jpg', 'IR638.jpg', 'IR626.png', 'IR638.png']
        vid3_invalid_rgb =  ['Vis118.jpg', 'Vis118.png']
        vid3_invalid_lwir = ['IR118.jpg', 'IR118.png']
        vid1_invalid_mrgb = ['VisForeground3586.jpg', 'VisForeground1014.jpg',
                             'VisForeground1300.jpg', 'VisForeground3748.jpg',
                             'VisForeground3586.png', 'VisForeground1014.png',
                             'VisForeground1300.png', 'VisForeground3748.png']
        vid1_invalid_mlwir = ['IRForeground3586.jpg', 'IRForeground1014.jpg'
                              'IRForeground1300.jpg', 'IRForeground3748.jpg',
                              'IRForeground3586.png', 'IRForeground1014.png'
                              'IRForeground1300.png', 'IRForeground3748.png']
        vid2_invalid_mrgb = ['VisForeground626.jpg', 'VisForeground638.jpg',
                             'VisForeground626.png', 'VisForeground638.png']
        vid2_invalid_mlwir = ['IRForeground626.jpg', 'IRForeground638.jpg',
                              'IRForeground626.png', 'IRForeground638.png']
        vid3_invalid_mrgb = ['VisForeground118.jpg', 'VisForeground118.png']
        vid3_invalid_mlwir = ['IRForeground118.jpg', 'IRForeground118.png']

        detectron_output = os.path.join(self.root, "DETECTRON-OUTPUT")
        litiv_output = os.path.join(detectron_output, "litiv2014")
        MASK_RCNN = False

        for video in videos:
            if video != 'vid2':
                if video == 'vid1':
                    for scene in vid1_scenes:
                        images = os.path.join(im_root, video, scene, 'videoFrames')

                        if MASK_RCNN:
                            masks = os.path.join(litiv_output, video, scene)
                        else:
                            masks = os.path.join(root, video, scene, 'Foreground')

                        rgb = [r for r in os.listdir(images) if r.startswith('Vis') and r not in vid1_invalid_rgb]
                        lwir = [lw for lw in os.listdir(images) if lw.startswith('IR') and lw not in vid1_invalid_lwir]
                        rgb = [os.path.join(images, r) for r in rgb]
                        rgb = misc.special_sort(rgb)
                        lwir = [os.path.join(images, lw) for lw in lwir]
                        lwir = misc.special_sort(lwir)
                        mrgb = [r for r in os.listdir(masks) if r.startswith('Vis') and r not in vid1_invalid_mrgb]
                        mlwir = [lw for lw in os.listdir(masks) if lw.startswith('IR') and lw not in vid1_invalid_mlwir]
                        mrgb = [os.path.join(masks, r) for r in mrgb]
                        mrgb = misc.special_sort(mrgb)
                        mlwir = [os.path.join(masks, lw) for lw in mlwir]
                        mlwir = misc.special_sort(mlwir)
                        disparity = os.path.join(root, video, scene, video + '_' + scene + '.txt')
                        disparity = [disparity for _ in range(len(rgb))]
                        # LITIV 2014 does not have a drange.txt, use a disparity file to fill its place.
                        drange = disparity
                        rgb_paths[video].extend(rgb)
                        lwir_paths[video].extend(lwir)
                        mask_rgb_paths[video].extend(mrgb)
                        mask_lwir_paths[video].extend(mlwir)
                        disparity_paths[video].extend(disparity)
                        drange_paths[video].extend(drange)
                else:
                    for scene in vid3_scenes:
                        images = os.path.join(im_root, video, scene, 'videoFrames')

                        if MASK_RCNN:
                            masks = os.path.join(litiv_output, video, scene)
                        else:
                            masks = os.path.join(root, video, scene, 'Foreground')

                        rgb = [r for r in os.listdir(images) if r.startswith('Vis') and r not in vid3_invalid_rgb]
                        lwir = [lw for lw in os.listdir(images) if lw.startswith('IR') and lw not in vid3_invalid_lwir]
                        rgb = [os.path.join(images, r) for r in rgb]
                        rgb = misc.special_sort(rgb)
                        lwir = [os.path.join(images, lw) for lw in lwir]
                        lwir = misc.special_sort(lwir)
                        mrgb = [r for r in os.listdir(masks) if r.startswith('Vis') and r not in vid3_invalid_mrgb]
                        mlwir = [lw for lw in os.listdir(masks) if lw.startswith('IR') and lw not in vid3_invalid_mlwir]
                        mrgb = [os.path.join(masks, r) for r in mrgb]
                        mrgb = misc.special_sort(mrgb)
                        mlwir = [os.path.join(masks, lw) for lw in mlwir]
                        mlwir = misc.special_sort(mlwir)
                        disparity = os.path.join(root, video, scene, video + '_' + scene + '.txt')
                        disparity = [disparity for _ in range(len(rgb))]
                        # LITIV 2014 does not have a drange.txt, use a disparity file to fill its place.
                        drange = disparity
                        rgb_paths[video].extend(rgb)
                        lwir_paths[video].extend(lwir)
                        mask_rgb_paths[video].extend(mrgb)
                        mask_lwir_paths[video].extend(mlwir)
                        disparity_paths[video].extend(disparity)
                        drange_paths[video].extend(drange)
            else:
                for cut in cuts:
                    if cut == 'cut1':
                        for scene in vid2_cut1_scenes:
                            images = os.path.join(im_root, video, cut, scene, 'videoFrames')

                            if MASK_RCNN:
                                masks = os.path.join(litiv_output, video, cut, scene)
                            else:
                                masks = os.path.join(root, video, cut, scene, 'Foreground')

                            rgb = [r for r in os.listdir(images) if r.startswith('Vis') and r not in vid2_invalid_rgb]
                            lwir = [lw for lw in os.listdir(images) if lw.startswith('IR')
                                    and lw not in vid2_invalid_lwir]
                            rgb = [os.path.join(images, r) for r in rgb]
                            rgb = misc.special_sort(rgb)
                            lwir = [os.path.join(images, lw) for lw in lwir]
                            lwir = misc.special_sort(lwir)
                            mrgb = [r for r in os.listdir(masks) if r.startswith('Vis') and r not in vid2_invalid_mrgb]
                            mlwir = [lw for lw in os.listdir(masks) if lw.startswith('IR')
                                     and lw not in vid2_invalid_mlwir]
                            mrgb = [os.path.join(masks, r) for r in mrgb]
                            mrgb = misc.special_sort(mrgb)
                            mlwir = [os.path.join(masks, lw) for lw in mlwir]
                            mlwir = misc.special_sort(mlwir)
                            disparity = os.path.join(root, video, cut, scene,
                                                     video + cut + '_' + scene + '.txt')
                            disparity = [disparity for _ in range(len(rgb))]
                            # LITIV 2014 does not have a drange.txt, use a disparity file to fill its place.
                            drange = disparity
                            rgb_paths[video].extend(rgb)
                            lwir_paths[video].extend(lwir)
                            mask_rgb_paths[video].extend(mrgb)
                            mask_lwir_paths[video].extend(mlwir)
                            disparity_paths[video].extend(disparity)
                            drange_paths[video].extend(drange)
                    else:
                        for scene in vid2_cut2_scenes:
                            images = os.path.join(im_root, video, cut, scene, 'videoFrames')

                            if MASK_RCNN:
                                masks = os.path.join(litiv_output, video, cut, scene)
                            else:
                                masks = os.path.join(root, video, cut, scene, 'Foreground')

                            rgb = [r for r in os.listdir(images) if r.startswith('Vis') and r not in vid2_invalid_rgb]
                            lwir = [lw for lw in os.listdir(images) if lw.startswith('IR')
                                    and lw not in vid2_invalid_lwir]
                            rgb = [os.path.join(images, r) for r in rgb]
                            rgb = misc.special_sort(rgb)
                            lwir = [os.path.join(images, lw) for lw in lwir]
                            lwir = misc.special_sort(lwir)
                            mrgb = [r for r in os.listdir(masks) if r.startswith('Vis') and r not in vid2_invalid_mrgb]
                            mlwir = [lw for lw in os.listdir(masks) if lw.startswith('IR')
                                     and lw not in vid2_invalid_mlwir]
                            mrgb = [os.path.join(masks, r) for r in mrgb]
                            mrgb = misc.special_sort(mrgb)
                            mlwir = [os.path.join(masks, lw) for lw in mlwir]
                            mlwir = misc.special_sort(mlwir)
                            disparity = os.path.join(root, video, cut, scene,
                                                     video + cut + '_' + scene + '.txt')
                            disparity = [disparity for _ in range(len(rgb))]
                            # LITIV 2014 does not have a drange.txt, use a disparity file to fill its place.
                            drange = disparity
                            rgb_paths[video].extend(rgb)
                            lwir_paths[video].extend(lwir)
                            mask_rgb_paths[video].extend(mrgb)
                            mask_lwir_paths[video].extend(mlwir)
                            disparity_paths[video].extend(disparity)
                            drange_paths[video].extend(drange)

        return self._reform(rgb_paths, lwir_paths, mask_rgb_paths, mask_lwir_paths, disparity_paths, drange_paths)
