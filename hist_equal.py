from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import copy
import cv2
import argparse

class Hist_Equal():
    def __init__(self):
        super(Hist_Equal, self).__init__()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--Input', type=str, default='img.jpg', help='the path of target image')
        self.parser.add_argument('--Output', type=str, default='res.jpg', help='the path of new image')
        
    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    
    def he_rgb(self, filename):
        #读取图像，存入数组中
        im = np.array(Image.open(filename))
        #获得各通道的值
        r = im[:, :, 0]
        g = im[:, :, 1]
        b = im[:, :, 2]

        #计算各通道直方图
        imhist_red,bins_red = np.histogram(r, 256, normed=True)
        imhist_gre,bins_gre = np.histogram(g, 256, normed=True)
        imhist_blu,bins_blu = np.histogram(b, 256, normed=True)

        #计算各通道累积分布函数
        cdf_red = imhist_red.cumsum()
        cdf_gre = imhist_gre.cumsum()
        cdf_blu = imhist_blu.cumsum()

        #累计函数归一化（由0～1变换至0~255）
        cdf_red = cdf_red * 255 / cdf_red[-1]
        cdf_gre = cdf_gre * 255 / cdf_gre[-1]
        cdf_blu = cdf_blu * 255 / cdf_blu[-1]

        #绘制直方图均衡化后的直方图
        equaled_red = np.interp(r.flatten(), bins_red[:256], cdf_red)
        equaled_gre = np.interp(g.flatten(), bins_gre[:256], cdf_gre)
        equaled_blu = np.interp(b.flatten(), bins_blu[:256], cdf_blu)

        #原始通道图
        im_red_source = r.reshape([im.shape[0], im.shape[1]])
        im_gre_source = g.reshape([im.shape[0], im.shape[1]])
        im_blu_source = b.reshape([im.shape[0], im.shape[1]])

        #均衡化后的通道图
        equaled_red = equaled_red.reshape([im.shape[0], im.shape[1]])
        equaled_gre = equaled_gre.reshape([im.shape[0], im.shape[1]])
        equaled_blu = equaled_blu.reshape([im.shape[0], im.shape[1]])

        #合并图像
        equaled_im = copy.deepcopy(im)
        equaled_im[:,:,0] = equaled_red
        equaled_im[:,:,1] = equaled_gre
        equaled_im[:,:,2] = equaled_blu
        return equaled_im

    ###以下是功能代码
    def get_size(self, filename):
        im = np.array(Image.open(filename))
        return im.shape[0],im.shape[1]

    def get_rgb(self, filename):
        im = np.array(Image.open(filename))
        r = im[:, :, 0]
        g = im[:, :, 1]
        b = im[:, :, 2]
        return r,g,b

    def get_hist(self, filename):
        r,g,b = get_rgb(filename)
        imhist_red, bins_red = np.histogram(r, 256, normed=True)
        imhist_gre, bins_gre = np.histogram(g, 256, normed=True)
        imhist_blu, bins_blu = np.histogram(b, 256, normed=True)
        return imhist_red,imhist_gre,imhist_blu,bins_red,bins_gre,bins_blu

    def draw_hist(self, filename):
        r,g,b =  get_rgb(filename)
        plt.subplot(131)
        plt.hist(r.flatten(),256)
        plt.subplot(132)
        plt.hist(g.flatten(),256)
        plt.subplot(133)
        plt.hist(b.flatten(),256)
        plt.show()

    def get_equaled(self, filename):
        r,g,b = get_rgb(filename)
        imhist_red,imhist_gre,imhist_blu,bins_red,bins_gre,bins_blu = get_hist(filename)
        #计算各通道累积分布函数
        cdf_red = imhist_red.cumsum()
        cdf_gre = imhist_gre.cumsum()
        cdf_blu = imhist_blu.cumsum()
        #累计函数归一化（由0～1变换至0~255）
        cdf_red = cdf_red * 255 / cdf_red[-1]
        cdf_gre = cdf_gre * 255 / cdf_gre[-1]
        cdf_blu = cdf_blu * 255 / cdf_blu[-1]

        equaled_red = np.interp(r.flatten(), bins_red[:256], cdf_red)
        equaled_gre = np.interp(g.flatten(), bins_gre[:256], cdf_gre)
        equaled_blu = np.interp(b.flatten(), bins_blu[:256], cdf_blu)

        return equaled_red,equaled_gre,equaled_blu

    def draw_equaled_hist(self, filename):
        equaled_red,equaled_gre,equaled_blu = get_equaled(filename)
        plt.figure()
        plt.subplot(131)
        plt.hist(equaled_red,256)
        plt.subplot(132)
        plt.hist(equaled_gre,256)
        plt.subplot(133)
        plt.hist(equaled_blu,256)
        plt.show()

    def get_ori_rgb(self, filename):
        r,g,b = get_rgb(filename)
        h,w = get_size(filename)
        im_red = r.reshape([h , w])
        im_gre = g.reshape([h , w])
        im_blu = b.reshape([h , w])
        return im_red,im_gre,im_blu

    def draw_rgb(self, im_r,im_g,im_b):
        #对于愿图像只需要get_ori_rgb即可，
        #对于均衡化之后的图像，get_equaled,然后equ_reshape一下
        plt.figure()
        plt.gray()
        plt.subplot(131)
        plt.imshow(im_r)
        plt.subplot(132)
        plt.imshow(im_g)
        plt.subplot(133)
        plt.imshow(im_b)
        plt.show()

    def equ_reshape(self, r,g,b,h,w):
        im_red = r.reshape([h , w])
        im_gre = g.reshape([h , w])
        im_blu = b.reshape([h , w])
        return im_red,im_gre,im_blu

    def equ_pic(self, filename):
        im = np.array(Image.open(filename))
        equaled_red,equaled_gre,equaled_blu = get_equaled(filename)
        h,w = get_size(filename)
        im_red,im_gre,im_blu = equ_reshape(equaled_red,equaled_gre,equaled_blu,h,w)
        im_copy = copy.deepcopy(im)
        im_copy[:,:,0] = im_red
        im_copy[:,:,1] = im_gre
        im_copy[:,:,2] = im_blu
        return im_copy

    def draw_pic(self, filename):
        im = equ_pic(filename)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(im)
        plt.show()

if __name__ == '__main__':
    hist_equal = Hist_Equal()
    opts = hist_equal.parse()
    _input = opts.Input
    _output = opts.Output
    res = Image.fromarray(np.uint8(hist_equal.he_rgb(_input)))
    res.save(_output)