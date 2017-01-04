import sys
sys.path.append( '../tools' )
import argparse
import FFDio
from FFDIter import FFDIter
import mxnet as mx
import numpy as np
import os
import config
from symbols.factory import SymbolFactory


def train_w_predetermined_data(args):
    dataiter_train = mx.io.ImageRecordIter(
      path_imgrec="../prepare_data/training_data.rec",
      data_shape=(3, 224, 224),
      path_imglist="../prepare_data/training_data.lst",
      label_width=136,
      mean_r = 128,
      mean_g = 128,
      mean_b = 128,
      batch_size = args.batch_size,
      label_name = 'label'
    )

    dataiter_test = mx.io.ImageRecordIter(
      path_imgrec="../prepare_data/test_data.rec",
      data_shape=(3, 224, 224),
      path_imglist="../prepare_data/test_data.lst",
      label_width=136,
      mean_r = 128,
      mean_g = 128,
      mean_b = 128,
      batch_size = args.batch_size,
      label_name = 'label'
    )

    # network
    symbol = SymbolFactory(args.save_prefix)
    network = symbol.get_symbol()

    model = mx.model.FeedForward(
        ctx = config._get_devs(**kwargs),
        symbol = network,       # network structure
        num_epoch = config._get_num_epoch(**kwargs),     # number of data passes for training 
        optimizer = mx.optimizer.Adam(learning_rate=1e-4),
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=0.05)
    )

    model.fit(
        X=dataiter_train,       # training data
        eval_data=dataiter_test, # validation data
        eval_metric = [NRMSE, 'rmse'],
        batch_end_callback = mx.callback.Speedometer(args.batch_size, 1), # output progress for each 200 data batches
        epoch_end_callback = mx.callback.do_checkpoint(args.save_prefix),
    )

    return model

def train_w_realtime_data(model):
    # make train data
    image_list_AFW = FFDio.collect_data_set('../original_images/AFW')
    image_list_IBUG = FFDio.collect_data_set('../original_images/IBUG')
    image_list_HELEN = FFDio.collect_data_set('../original_images/HELEN/trainset')
    image_list_LFPW = FFDio.collect_data_set('../original_images/LFPW/trainset')
    image_list_train = image_list_AFW + image_list_HELEN + image_list_LFPW

    # make validation data
    '''
    image_list_indoor = FFDio.collect_data_set('../original_images/300W/01_Indoor')
    image_list_outdoor = FFDio.collect_data_set('../original_images/300W/02_Outdoor')
    image_list_val = image_list_indoor + image_list_outdoor
    '''
    iter_val = mx.io.ImageRecordIter(
      path_imgrec="../prepare_data/test_data.rec",
      data_shape=(3, 224, 224),
      path_imglist="../prepare_data/test_data.lst",
      label_width=136,
      mean_r = 128,
      mean_g = 128,
      mean_b = 128,
      batch_size = args.batch_size,
      label_name = 'label'
    )



    # make iterator
    iter_train = FFDIter(image_list_train, args.batch_size, True, 224, 224)
    #iter_val = FFDIter(image_list_val, args.batch_size, False, 224, 224)
    
    model.fit(
        X = iter_train,       # training data
        eval_data = iter_val, # validation data
        eval_metric = NRMSE, #'rmse',
        batch_end_callback = mx.callback.Speedometer(args.batch_size, 1), # output progress for each 200 data batches
        epoch_end_callback = mx.callback.do_checkpoint(args.save_prefix + '_finetuned'),
    ) 


def NRMSE(label, pred):
    batch_size = label.shape[0]

    total_sum = 0
    for i in range(0, batch_size):
        gt = label[i, :]
        gt = np.reshape(gt, (2, 68))
        
        pts = pred[i, :]
        pts = np.reshape(pts, (2, 68))
        iod = np.linalg.norm( gt[:, 36] - gt[:, 45] )

        sum = 0
        for i in range(0, 68):
            sum += np.linalg.norm( gt[:, i] - pts[:, i] )
        rmse_68 = sum / (68 * iod)
        total_sum += rmse_68
    
    return total_sum / batch_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MXNet Trainer')
    parser.add_argument('--save-prefix', type=str, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--gpus', type=str, required=False)
    parser.add_argument('--num-epoch', type=int, required=True)

    args = parser.parse_args()   
    kwargs = args.__dict__
    config._set_logger(**kwargs)

    model = train_w_predetermined_data(args)
    
    '''
    # load pretrained model
    sym, arg_params, aux_params = mx.model.load_checkpoint('vgg_16_reduced', 10)
    model = mx.model.FeedForward(
        ctx = config._get_devs(**kwargs),
        symbol = sym,       # network structure
        num_epoch = config._get_num_epoch(**kwargs),     # number of data passes for training 
        optimizer = mx.optimizer.Adam(learning_rate=1e-4),
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=0.05),
        arg_params = arg_params,
        aux_params = aux_params
    )
    '''

    model = train_w_realtime_data(model)
    
