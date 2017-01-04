import json
import time
import requests
import cv2
import numpy as np
import math
import os
import shutil
import threading
import Queue
import glob
import random
import copy

import sys
sys.path.append( '../tools' )
import FFDio


norm_width = 224
norm_height = 224

class CropThread(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            try:
                dst_dir, image_name, is_train = self.queue.get()
            except Queue.Empty:
                break

            # image
            img_cv = cv2.imread(image_name)

            # gt
            gt_name = image_name[0:image_name.rfind('.')] + '.pts'
            pts = FFDio.load_gt(gt_name)
            
            if is_train:
                norm_imgs = FFDio.crop(img_cv, pts, True, 10, norm_width, norm_height)
            else:
                norm_imgs = []
                norm_img, norm_pts = FFDio.crop(img_cv, pts, False, 1, norm_width, norm_height)
                norm_imgs.append( (norm_img, norm_pts) )  

            for random_index, img_info in enumerate(norm_imgs):
                norm_img = img_info[0]
                norm_gt = img_info[1]
                
                # image
                output_name = '%s/%s_%03d.png' % (dst_dir, image_name.split('/')[-1].split('.')[0], random_index)
                cv2.imwrite(output_name, norm_img)

                # gt
                output_name = '%s/%s_%03d.pts' % (dst_dir, image_name.split('/')[-1].split('.')[0], random_index)
                FFDio.save_gt(output_name, norm_gt)


            self.queue.task_done()




def crop_dir(src_dir, dst_dir, is_train=True):

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)


    queue = Queue.Queue()
    for i in range(1):
        t = CropThread(queue)
        t.setDaemon(True)
        t.start()

    image_list = FFDio.collect_data_set(src_dir)

    for image_name in image_list:
        queue.put((dst_dir, image_name, is_train))
    queue.join()
    print "finished : " + src_dir



if __name__ == '__main__':

    src_image_root = '../original_images'
    dst_image_root = '../crop_images'
    train_set = [('AFW', 'AFW'), ('HELEN/trainset', 'HELEN_trainset'), ('IBUG', 'IBUG'), ('LFPW/trainset', 'LFPW_trainset')]
    test_set = [('300W/01_Indoor', '300W_Indoor'), ('300W/02_Outdoor', '300W_Outdoor')]
    

    if not os.path.exists(dst_image_root):
        os.mkdir(dst_image_root) 
    
    # training data
    for src_dir, dst_dir in train_set:
        src_dir = src_image_root + '/' + src_dir
        dst_dir = dst_image_root + '/' + dst_dir
        if os.path.isdir(src_dir):
            crop_dir(src_dir, dst_dir, True)
    
    # test data
    for src_dir, dst_dir in test_set:
        src_dir = src_image_root + '/' + src_dir
        dst_dir = dst_image_root + '/' + dst_dir
        if os.path.isdir(src_dir):
            crop_dir(src_dir, dst_dir, False)


    # make test list for mxnet
    list = []
    index = 0
    for _, dst_dir in test_set:
        dst_dir = dst_image_root + '/' + dst_dir
        image_list = FFDio.collect_data_set(dst_dir)
        for image_name in image_list:
            image = cv2.imread(image_name)
            gt_name = image_name[0:image_name.rfind('.')] + '.pts'
            pts = FFDio.load_gt(gt_name)

            pts[0, :] = pts[0, :] / image.shape[1]
            pts[1, :] = pts[1, :] / image.shape[0]
            pts = pts.flatten()

            list.append( (index, image_name, pts) )
            index += 1
    random.shuffle(list)

    f = open('test_data.lst', 'wt')
    for image_info in list:
        f.write('%d ' % image_info[0])
        for i in range(0, 136):
            f.write('%f ' % image_info[2][i])
        f.write('%s\n' % image_info[1])
    f.close()


    # make training list for mxnet
    list = []
    index = 0
    for _, dst_dir in train_set:
        dst_dir = dst_image_root + '/' + dst_dir
        image_list = FFDio.collect_data_set(dst_dir)
        for image_name in image_list:
            image = cv2.imread(image_name)
            gt_name = image_name[0:image_name.rfind('.')] + '.pts'
            pts = FFDio.load_gt(gt_name)

            pts[0, :] = pts[0, :] / image.shape[1]
            pts[1, :] = pts[1, :] / image.shape[0]
            pts = pts.flatten()

            list.append( (index, image_name, pts) )
            index += 1
    random.shuffle(list)

    f = open('training_data.lst', 'wt')
    for image_info in list:
        f.write('%d ' % image_info[0])
        for i in range(0, 136):
            f.write('%f ' % image_info[2][i])
        f.write('%s\n' % image_info[1])
    f.close()
    
    
            
 
