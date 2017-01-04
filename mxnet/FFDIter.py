import mxnet as mx
import numpy as np
import cv2
import sys
sys.path.append( '../tools' )
import FFDio
import random

class FFDIter(mx.io.DataIter):
    """
    Detection Iterator, which will feed data and label to network
    Optional data augmentation is performed when providing batch
    Parameters:
    ----------
    imdb : Imdb
        image database
    batch_size : int
        batch size
    data_shape : int or (int, int)
        image shape to be resized
    mean_pixels : float or float list
        [R, G, B], mean pixel values
    rand_samplers : list
        random cropping sampler list, if not specified, will
        use original image only
    rand_mirror : bool
        whether to randomly mirror input images, default False
    shuffle : bool
        whether to shuffle initial image list, default False
    rand_seed : int or None
        whether to use fixed random seed, default None
    max_crop_trial : bool
        if random crop is enabled, defines the maximum trial time
        if trial exceed this number, will give up cropping
    is_train : bool
        whether in training phase, default True, if False, labels might
        be ignored
    """

    def __init__(self, imdb, batch_size, \
                 aug, norm_width, norm_height):
        super(FFDIter, self).__init__()

        self._imdb = imdb
        self.batch_size = batch_size

        self._current = 0
        self._size = len(imdb)
        self._index = np.arange(self._size)
        self._aug = aug
        self._data = None
        self._label = None
        self._norm_width = norm_width
        self._norm_height = norm_height
        self._get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in self._data.items()]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in self._label.items()]

    def reset(self):
        self._current = 0
        random.shuffle(self._index)

    def iter_next(self):
        return self._current < self._size

    def next(self):
        if self.iter_next():
            self._get_batch()
            data_batch = mx.io.DataBatch(data=self._data.values(),
                                   label=self._label.values(),
                                   pad=self.getpad(), index=self.getindex())
            self._current += self.batch_size
            return data_batch
        else:
            raise StopIteration

    def getindex(self):
        return self._current / self.batch_size

    def getpad(self):
        pad = self._current + self.batch_size - self._size
        return 0 if pad < 0 else pad

    def _get_batch(self):
        """
        Load data/label from dataset
        """
        batch_data = []
        batch_label = []
        for i in range(self.batch_size):
            if (self._current + i) >= self._size:
                # use padding from middle in each epoch
                idx = (self._current + i + self._size / 2) % self._size
                index = self._index[idx]
            else:
                index = self._index[self._current + i]
            # index = self.debug_index
            im_path = self._imdb[index]
            img = cv2.imread(im_path)
            gt_name = im_path[0:im_path.rfind('.')] + '.pts'
            gt = FFDio.load_gt(gt_name)
            data, label = self._data_augmentation(img, gt, self._aug, self._norm_width, self._norm_width)

            # swap channel
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

            # normalize
            data = data.astype(np.float32)
            data -= [128, 128, 128]
            #data = data / 128
            label[0, :] = label[0, :] / data.shape[1]
            label[1, :] = label[1, :] / data.shape[0]
            
            label = label.flatten()
            data = data.transpose((2, 0, 1))
            #data = np.swapaxes(data, 0, 2)
            

            batch_data.append(data)
            batch_label.append(label)
        # pad data if not fully occupied
        for i in range(self.batch_size - len(batch_data)):
            assert len(batch_data) > 0
            batch_data.append(batch_data[0] * 0)
        self._data = {'data': mx.nd.array(np.array(batch_data))}
        self._label = {'label': mx.nd.array(np.array(batch_label))}

    def _data_augmentation(self, data, label, aug, norm_width, norm_height):
        """
        perform data augmentations: crop, mirror, resize, sub mean, swap channels...
        """
        return FFDio.crop(img=data, pts=label, bool_aug=aug, norm_width=norm_width, norm_height=norm_height)

if __name__ == '__main__':

    image_list_AFW = FFDio.collect_data_set('../original_images/AFW')
    image_list_IBUG = FFDio.collect_data_set('../original_images/IBUG')
    image_list_HELEN = FFDio.collect_data_set('../original_images/HELEN')
    image_list = image_list_AFW + image_list_HELEN + image_list_IBUG

    print 'collected %d images' % len(image_list)

    batch_size = 10
    train_iter = FFDIter(image_list, batch_size, False, 224, 224)
    data_batch = train_iter.next()

    data = data_batch.data
    label = data_batch.label
    print data[0]
    for i in range(0, batch_size):
        image = data[0][i].asnumpy()
        pts = label[0][i].asnumpy()
        print pts.shape
        image = image.transpose((1, 2, 0))
        image += [128, 128, 128]
        image = image.astype('uint8')
        image = image.copy()

        for j in range(0, 68):
            cv2.circle(image, (int(pts[j] * image.shape[1]), int(pts[j + 68] * image.shape[0])), 2, (255, 0, 0), 2)

        cv2.imshow('temp', image)
        cv2.waitKey(-1)
