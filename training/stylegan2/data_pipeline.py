import os
import os.path as osp
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import flax
import numpy as np
from PIL import Image
import os
from typing import Sequence
from tqdm import tqdm
import json
from tqdm import tqdm


def prefetch(dataset, n_prefetch):
    # Taken from: https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py
    ds_iter = iter(dataset)
    ds_iter = map(lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x),
                  ds_iter)
    if n_prefetch:
        ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
    return ds_iter


def get_data(data_dir, img_size, img_channels, num_classes, num_devices, batch_size, shuffle_buffer=1000):
    """

    Args:
        data_dir (str): Root directory of the dataset.
        img_size (int): Image size for training.
        img_channels (int): Number of image channels.
        num_classes (int): Number of classes, 0 for no classes.
        num_devices (int): Number of devices.
        batch_size (int): Batch size (per device).
        shuffle_buffer (int): Buffer used for shuffling the dataset.

    Returns:
        (tf.data.Dataset): Dataset.
    """

    def pre_process(features):
        image = features['image']
        image = tf.cast(image, dtype='float32')
        # image = tf.image.resize(image, size=[img_size, img_size], method='bicubic', antialias=True)
        image = tf.image.random_flip_left_right(image)
        image = (image - 127.5) / 127.5
        return {'image': image}

    def shard(data):
        # Reshape images from [num_devices * batch_size, H, W, C] to [num_devices, batch_size, H, W, C]
        # because the first dimension will be mapped across devices using jax.pmap
        data['image'] = tf.reshape(data['image'], [num_devices, -1, img_size, img_size, img_channels])
        return data

    dataset_info = {'num_examples': 70000}
    split = tfds.split_for_jax_process('train')
    ds = tfds.load(osp.basename(data_dir), split=split, data_dir=osp.dirname(data_dir))
    
    ds = ds.shuffle(min(dataset_info['num_examples'], shuffle_buffer))
    ds = ds.map(pre_process, tf.data.AUTOTUNE)
    ds = ds.batch(batch_size * num_devices, drop_remainder=True)
    ds = ds.map(shard, tf.data.AUTOTUNE)
    ds = ds.prefetch(1)
    return ds, dataset_info



