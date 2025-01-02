import random, re
import numpy as np
import torch.utils.data as data
import utils.utils_image as util


# 提取数字的函数
def extract_number_lr(s):
    match = re.search(r'/X2/(\d+)X2\.png$', s)
    return int(match.group(1)) if match else -1

# 提取数字的函数
def extract_number(reg_str, s):
    match = re.search(reg_str, s)
    return int(match.group(1)) if match else -1


class DatasetSR(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])
        
        # 构建正则表达式
        reg_hr = rf"/HR/(\d+)\.bmp$"  # 使用 rf 字符串，reg_str 已经是原始字符串
        reg_lr = rf"/X{self.sf}/(\d+)X{self.sf}\.png$"

        # 对数据排个序，从而成为图像对 pair
        self.paths_H = sorted(self.paths_H, key=lambda pth: extract_number(reg_hr, pth))
        self.paths_L = sorted(self.paths_L, key=lambda pth: extract_number(reg_lr, pth))

        #上面這兩個就是image來源路徑
        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        #dataloader以index為input呼叫__getitem__(index)
        #利用index找出對應圖片的檔案路徑到H_path
        img_H = util.imread_uint(H_path, self.n_channels)
        #讀進image
        img_H = util.uint2single(img_H)
        # uint2single : return np.float32(img/255.), 歸一化到0~1之間

        # ------------------------------------
        # modcrop
        # ------------------------------------
        #把High resolution image的長和寬切割成scale的倍數(不然1037要怎麼從別人x2過來)该函数常用于图像超分辨率任务中，以确保图像在缩放操作后不会产生尺寸不匹配的问题。
        img_H = util.modcrop(img_H, self.sf)

        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.paths_L:
            # --------------------------------
            # directly load L image
            # --------------------------------
            L_path = self.paths_L[index]
            img_L = util.imread_uint(L_path, self.n_channels)
            img_L = util.uint2single(img_L)

        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_H.shape[:2]
            img_L = util.imresize_np(img_H, 1 / self.sf, True)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
