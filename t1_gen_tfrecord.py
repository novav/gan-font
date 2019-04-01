import os
from PIL import Image
import numpy as np
import tensorflow as tf

DATA_SET_DIR = 'O:/AI/train/'

GEN_FILE_DIR = './tfrecords/'

def get_files_labels(data_dir):
    photos = []
    names = []
    i = 0
    for filename in os.listdir(DATA_SET_DIR):
        path = os.path.join(data_dir, filename)
        if os.path.isdir(path):
            print('++' * 8, path)
            for ff in os.listdir(path):
                fs = os.path.join(path, ff)
                names.append(ff)
                photos.append(fs)
                i += 1
    print('', photos[39987])
    print('---', photos[39988])
    print('', photos[39989])
    print('', i)
    return photos, names

def byte_feature(value):
    # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def image_to_tfexample(image_data, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': byte_feature(image_data),
        'label': byte_feature(label)
    }))

def _conert_dataset(split_name, file_names, labels):

    output_file = os.path.join(GEN_FILE_DIR, split_name + '.tfrecord')

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    output_writer = tf.python_io.TFRecordWriter(output_file, options=options)

    len_size = len(file_names)
    for i, (filename, label) in enumerate(zip(file_names, names)):

        image_data = Image.open(filename)
        image_data = image_data.resize((224, 224))
        image_data = np.array(image_data.convert('L'))
        image_data = image_data.tobytes()

        label = bytes(label, encoding='utf-8')

        example = image_to_tfexample(image_data, label)
        output_writer.write(example.SerializeToString())

        if i % 100 == 0 :
            print('----------', i/len_size, i, len_size)


pp, names = get_files_labels(DATA_SET_DIR)

# _conert_dataset('train_2', pp, names)
# filename = pp[-1]
# print(filename)
# image_data = Image.open(filename)
# print(image_data)
