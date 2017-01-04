import sys
sys.path.append( '../tools' )
import FFDio
from FFDIter import FFDIter
import mxnet as mx
import numpy as np
import os
import cv2

if __name__ == '__main__':
    batch_size = 10
    
    # make validation data
    image_list_indoor = FFDio.collect_data_set('../original_images/300W/01_Indoor')
    image_list_outdoor = FFDio.collect_data_set('../original_images/300W/02_Outdoor')
    image_list_val = image_list_indoor + image_list_outdoor

    # make iterator
    
    iter_val = FFDIter(image_list_val, batch_size, False, 224, 224)
    


    prefix = 'vgg_16_reduced'
    model = mx.model.FeedForward.load(prefix, 50)

    data_batch = iter_val.next()
    data = data_batch.data
    label = data_batch.label
    print data[0].shape

    pred_loc = model.predict(data[0])
    

    for i in range(0, batch_size):
        image = data[0][i].asnumpy()
        pts = pred_loc[i]


        image = image.transpose((1, 2, 0))
        image += [128, 128, 128]
        image = image.astype('uint8')
        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for j in range(0, 68):
            cv2.circle(image, (int(pts[j] * image.shape[1]), int(pts[j + 68] * image.shape[0])), 2, (255, 0, 0), 2)

        cv2.imshow('temp', image)
        cv2.waitKey(-1)
