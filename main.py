import os
import numpy as np
import glob
import os.path as osp
import matplotlib.pyplot as plt
from skimage import io
import matplotlib.patches as patches
import torch
from utils.deal_config import read_config
from tracker_manager import TrackerManager
from torchvision import transforms as T
import time
from torch.cuda.amp import autocast


if __name__ == '__main__':
    config = read_config()

    if config.display:
        colors = np.random.rand(32, 3)
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    load_time = 0
    for path in glob.glob(config.train_parttern):
        name = path.split('/')[-3]
        print(f"正在处理: {name}")
        manager = TrackerManager(config.max_feature, config.max_ttl, config.min_age, config.threshold, config.tolerate_range, config.distance_base_cost)

        # 加载数据 [frame, _, x1, y1, w, h, _, _, _]
        data = np.loadtxt(path, delimiter=',')

        torch.cuda.synchronize()
        st = time.time()
        with open(osp.join(config.out_dir, f'{name}.txt'), 'w') as out_file:
            for frame in range(1, int(data[:, 0].max())):
                if config.display:
                    fn = os.path.join(config.test_dir, name, 'img1', f'{frame:06d}.jpg')
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(name)

                # 取一帧的信息:[[x1, y1, x2, y2], [x1, y1, x2, y2]...]
                detections = data[data[:, 0]==frame, 2:6]

                # 更新结果
                boxes = manager.update(detections, im)

                # 输出到文件，格式: [[frame, x, y, w, h], [frame, x, y, w, h]...]
                for box in boxes:
                    if config.display:
                        box = box.astype(np.int32)
                        ax1.add_patch(patches.Rectangle(
                            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                            fill=False, lw=3, ec= colors[box[4]%32,:]))
                            
                        ax1.add_patch(patches.Rectangle(
                            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                            fill=False, lw=3, ec= colors[box[4]%32,:]))

                if config.display:
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

        torch.cuda.synchronize()
        print((time.time() - st)/int(data[:, 0].max()))
