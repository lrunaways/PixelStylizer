import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

TFRECORDS_DIRECTORY_PATH = "D:\data\TFRecords-baseline"
NPY_OUT_PATH = "D:\data\pixelation"
feature_description = {
    'patch_compressed': tf.io.FixedLenFeature([], tf.string),
    'patch_clean': tf.io.FixedLenFeature([], tf.string),
}

@tf.function
def parse(x):
    x = tf.io.parse_single_example(x, feature_description)
    x = tf.stack([tf.io.parse_tensor(x['patch_compressed'], tf.dtypes.uint8),\
                  tf.io.parse_tensor(x['patch_clean'], tf.dtypes.uint8)])
    # x = tf.dtypes.cast(x, tf.float32)
    # x = ((x - tf.math.reduce_min(x, axis=0, keepdims=True)) /
    #      (tf.math.reduce_max(x, axis=0, keepdims=True) - tf.math.reduce_min(x, axis=0, keepdims=True)))

    return x


if __name__ == "__main__":
    filenames = [os.path.join(TFRECORDS_DIRECTORY_PATH, fn) for fn in next(os.walk(TFRECORDS_DIRECTORY_PATH))[2]]
    dataset = tf.data.TFRecordDataset(filenames, compression_type='ZLIB')
    dataset = dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for i, x in enumerate(dataset):
        x = x.numpy()
        np.save(os.path.join(NPY_OUT_PATH, f"{i}.npy"), x)
    print(1)