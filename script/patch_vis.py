import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

path = "../sample_tensor"



for i in range(0,800,100):
    cur_rgb = torch.load(os.path.join(path, "rgb_{0}.txt".format(i)))[0].permute(1, 2, 0)
    np_rgb = np.int8((np.array(cur_rgb)*128)+128)
    print(np.array(cur_rgb))
    print((np.array(cur_rgb)*128)+128)

    print(np.amax(np.array(cur_rgb)))
    print(np.amin(np.array(cur_rgb)))

    cur_lwir = torch.load(os.path.join(path, "lwir_{0}.txt".format(i)))
    print(np_rgb.shape)

    plt.imshow(np_rgb)
    plt.show()
    #img.save('my.png')
    # img.save("test.png")
    # print(cur_rgb[0].shape)
    # saved = Image.open("my.png")
    # print(np.asarray(saved))