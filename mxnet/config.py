import argparse
import mxnet as mx
import numpy as np
import logging
import coloredlogs
import os

def _get_devs(**kwargs):
    gpus = kwargs.get('gpus', None)
    devs = [mx.cpu()] if gpus is None else [mx.gpu(int(x)) for x in gpus.split(',')]
    return devs


def get_kvstore(**kwargs):
    kv = mx.kvstore.create(kwargs.get('kv_store', 'local'))
    devs = _get_devs(**kwargs)
    if 'local' in kv.type and len(devs) == 1:
        kv = None
    return kv

def _get_num_epoch(**kwargs):
    return int(kwargs.get('num_epoch', 50))

def _set_logger(**kwargs):
    kv = get_kvstore(**kwargs)
    logging_format = '%(asctime)-15s Node[' + str(kv.rank if kv else 0) + '] %(message)s'
    save_prefix = kwargs.get('save_prefix')
    save_dir = os.path.abspath(os.path.dirname(save_prefix))
    log_path = save_prefix + '.log'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = logging.getLogger()
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(logging_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    coloredlogs.install(level='DEBUG', fmt=logging_format)
    logger.info('START WITH ARGUMENTS %s', kwargs)
