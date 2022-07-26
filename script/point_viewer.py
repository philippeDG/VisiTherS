"""
implementation for some visual disparity points.

author: Philippe Duplessis-Guindon
date: 2022-07-26
"""

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "../../../datasets/DATASET-COPY/LITIV2018/vid07/"
MASK_PATH = "/home/phil/master/datasets/DETECTRON-OUTPUT/litiv2018/rectified/vid07/"



DISPARITY   = os.path.join(DATA_PATH, "disparity")
RGB        = os.path.join(DATA_PATH, "rgb")
LWIR        = os.path.join(DATA_PATH, "lwir")

dis_file = open(os.path.join(DISPARITY, "0.txt"))


rgb_pic = os.path.join(RGB, "0.png")
lwir_pic = os.path.join(LWIR, "0.png")
rgb_pic_mask = os.path.join(MASK_PATH, "rgb/00551.png")
lwir_pic_mask = os.path.join(MASK_PATH, "lwir/00551.png")

print(rgb_pic)

def print_patch_block(im_path, patch_size, x, y, show=False):
    im = Image.open(im_path)
    arr = np.asarray(im)

    mod = arr[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size]

    if show:
        print('x, y: ({0}, {1})'.format(x,y) )

        plt.imshow(mod, cmap="gray")
        plt.show()
    im.close()

def print_patch_precise(im_path, patch_size, x, y, show=False):
    im = Image.open(im_path)
    arr = np.asarray(im)

    mod = arr[y : y + patch_size, x : x + patch_size]

    if show:
        print('x, y: ({0}, {1})'.format(x,y) )
        plt.imshow(mod)
        plt.show()
    im.close()


def image_patch_stats(im_path, patch_size):
    im = Image.open(im_path)
    arr = np.asarray(im)

    y, x, _ = arr.shape

    print("patch_id_x, patch_id_y: ({0},{1})".format(x/patch_size, y/patch_size))

###
# Returns the disparity file in an int array in the format (y_red, x, y_lwir)
###
def get_disparities(path):
    dis_file = open(path)
    line = dis_file.readline()
    disparities = []
    while line:
        disparities.append([int(d) for d in line.split("\n")[0].split(" ")])
        line = dis_file.readline()
    dis_file.close()

    return np.array(disparities)


###
# Overlaying every point on the disparity file on the corresponding image
###
def trace_disparities(pic_id):
    disp = get_disparities(os.path.join(DISPARITY, "{}.txt".format(pic_id)))
    rgb_img = Image.open(os.path.join(RGB, "{}.png".format(pic_id)))
    lwir_img = Image.open(os.path.join(LWIR, "{}.png".format(pic_id)))

    rgb_arr = np.asarray(rgb_img)
    lwir_arr = np.asarray(lwir_img)

    plt.imshow(rgb_arr)
    print(disp[0][1])
    plt.scatter(disp[:,0], disp[:,1], marker="x", color="green", s=20)
    plt.show()
    plt.imshow(lwir_arr)
    plt.scatter(disp[:,2], disp[:,1], marker="x", color="red", s=20)
    plt.show()

patch_size = 36
x = 8
y = 5

image_patch_stats(rgb_pic, patch_size)
print_patch_block(rgb_pic_mask, patch_size, x, y, True)
print_patch_block(rgb_pic, patch_size, x, y, True)
print_patch_block(lwir_pic, patch_size, x-1, y, True)
print_patch_block(lwir_pic_mask, patch_size, x-1, y, True)

trace_disparities(0)