import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import DataParallel
import cv2
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import sys

from bbox_model.config import Config
from bbox_model.kpda_parser import KPDA
from bbox_model.cascade_pyramid_network import CascadePyramidNet
from bbox_model.utils import draw_keypoints
from bbox_model.helper.keypoint_encoder import KeypointEncoder
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pandas as pd


def compute_keypoints(config, img0, net, encoder, doflip=False):
    img_h, img_w, _ = img0.shape
    # min size resizing
    scale = config.img_max_size / (max(img_w, img_h)*1.0)
    img_h2 = int(img_h * scale)
    img_w2 = int(img_w * scale)
    img = cv2.resize(img0, (img_w2, img_h2), interpolation=cv2.INTER_CUBIC)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # channel, height, width
    img[[0, 2]] = img[[2, 0]]
    img = img / 255.0
    img = (img - config.mu) / (config.sigma*1.0)
    pad_imgs = np.zeros([1, 3, config.img_max_size, config.img_max_size], dtype=np.float32)
    pad_imgs[0, :, :img_h2, :img_w2] = img
    data = torch.from_numpy(pad_imgs)
    data = data.cuda(async=True)
    _, hm_pred = net(data)
    hm_pred = F.relu(hm_pred, False)
    hm_pred = hm_pred[0].data.cpu().numpy()

    if doflip:
        a = np.zeros_like(hm_pred)
        a[:, :, :img_w2 // config.hm_stride] = np.flip(hm_pred[:, :, :img_w2 // config.hm_stride], 2)
        for conj in config.conjug:
            a[conj] = a[conj[::-1]]
        hm_pred = a

    return hm_pred

    # x, y = encoder.decode_np(hm_pred, scale, config.hm_stride, method='maxoffset')
    # keypoints = np.stack([x, y, np.ones(x.shape)], axis=1).astype(np.int16)
    # return keypoints

def pts2bbox(pts):
    locs = np.array(pts)
    minimums = np.amin(locs, axis=0)
    maximums = np.amax(locs, axis=0)
    bbox = np.array([[max(minimums[0], 0), max(minimums[1], 0),
                      maximums[0], maximums[1]]], dtype=np.int64)
    return bbox[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--clothes', help='specify the clothing type', default='whale')
    parser.add_argument('-g', '--gpu', help='cuda device to use', default='0')
    parser.add_argument('-m', '--model', help='specify the model', default=None)
    parser.add_argument('-v', '--vis', help='whether visualize result', default=False)

    args = parser.parse_args(sys.argv[1:])
    config = Config(args.clothes)

    mode = 'train'

    if mode == 'train':
        val_kpda = KPDA(config,'test_train')
    else:
        val_kpda = KPDA(config, 'test')

    args.model = r'NONE'
    name = r'./results/se50_flip_' + mode

    print('Testing: ' + config.clothes)
    print('Validation sample number: %d' % val_kpda.size())
    net = CascadePyramidNet(config)
    checkpoint = torch.load(args.model)  # must before cuda
    net.load_state_dict(checkpoint['state_dict'])
    net = net.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)
    net.eval()
    encoder = KeypointEncoder()
    nes = []

    img_id = []
    keypoint_word = []

    Image, x0, y0, x1, y1 = [],[],[],[],[]

    for idx in tqdm(range(val_kpda.size())):
        img_path = val_kpda.get_image_path(idx)
        img0 = cv2.imread(img_path)  #X
        img0_flip = cv2.flip(img0, 1)
        img_h, img_w, _ = img0.shape

        # mask_path = img_path.replace('Images', 'Masks').replace('.jpg', '.npy')
        # if os.path.exists(mask_path):
        #     mask = np.load(mask_path)
        #     mask_h, mask_w = mask.shape
        #     assert mask_h == img_h and mask_w == img_w
        #     mask_scale = config.img_max_size / config.hm_stride / max(img_w, img_h)
        #     mask_h2 = int(mask_h * mask_scale)
        #     mask_w2 = int(mask_w * mask_scale)
        #     mask = cv2.resize(mask, (mask_w2, mask_h2))
        #     pad_mask = np.zeros([config.img_max_size//config.hm_stride, config.img_max_size//config.hm_stride], dtype=mask.dtype)
        #     pad_mask[:mask_h2, :mask_w2] = mask
        #     pad_mask = binary_dilation(pad_mask, iterations=10).astype(np.float32)
        # else:
        #     pad_mask = np.ones([config.img_max_size//config.hm_stride, config.img_max_size//config.hm_stride], dtype=np.float32)


# ----------------------------------------------------------------------------------------------------------------------
        scale = config.img_max_size / (max(img_w, img_h)*1.0)
        with torch.no_grad():
            hm_pred = compute_keypoints(config, img0, net, encoder) #* pad_mask
            hm_pred2 = compute_keypoints(config, img0_flip, net, encoder, doflip=True) #* pad_mask

        x, y = encoder.decode_np(hm_pred + hm_pred2, scale, config.hm_stride, (img_w/2, img_h/2), method='maxoffset')
        # x, y = encoder.decode_np(hm_pred2 , scale, config.hm_stride, (img_w / 2, img_h / 2),
        #                          method='maxoffset')
        keypoints = np.stack([x, y, np.ones(x.shape)], axis=1).astype(np.int16)
        bbox = pts2bbox(keypoints)
        # print(keypoints)
        # print(bbox)
# ----------------------------------------------------------------------------------------------------------------------
        # keypoints = compute_keypoints(config, img0, net, encoder)
        # keypoints_flip = compute_keypoints(config, img0_flip, net, encoder)
        # keypoints_flip[:, 0] = img0.shape[1] - keypoints_flip[:, 0]
        # for conj in config.conjug:
        #     keypoints_flip[conj] = keypoints_flip[conj[::-1]]
        # keypoints2 = np.copy(keypoints)
        # keypoints2[:, :2] = (keypoints[:, :2] + keypoints_flip[:, :2]) // 2
# ----------------------------------------------------------------------------------------------------------------------
        if args.vis:
            # kp_img = draw_keypoints(img0, keypoints[2:3])
            # kp_img = draw_keypoints(img0, keypoints[7:8])
            kp_img = draw_keypoints(img0, keypoints)
            cv2.rectangle(kp_img, (bbox[0],bbox[1]),(bbox[2],bbox[3]), (0,255,0), 5)
            cv2.imwrite( './tmp/{0}{1}.png'.format(config.clothes, idx), kp_img)

        # break

        img_id.append(os.path.split(img_path)[1])

        pt_str = r''
        for i in range(10):
            x,y,v = keypoints[i]
            pt = str(x)+'_'+str(y)+' '
            pt_str += pt

        keypoint_word.append(pt_str)

        # break
        # left, right = config.datum
        # x1, y1, v1 = kpts[left]
        # x2, y2, v2 = kpts[right]
        # if v1 == -1 or v2 == -1:
        #     continue
        # width = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        # ne = normalized_error(keypoints, kpts, width)
        # nes.append([ne])

        x0_, y0_, x1_, y1_ = bbox
        x0.append(x0_)
        y0.append(y0_)
        x1.append(x1_)
        y1.append(y1_)

    pd.DataFrame({'Image':img_id, 'x0':x0, 'y0':y0, 'x1':x1, 'y1':y1}).to_csv( '{}.csv'.format(name+'_bbox'), index=None)
    pd.DataFrame({'key_id':img_id, 'word':keypoint_word}).to_csv( '{}.csv'.format(name+'_keypoint'), index=None)



