# Written by Jing Chen
# Data generated from simulator has the following format:
#     - Directory_name
#         - driving_log.csv     (Contains image paths and their corresponding angle, throttle)
#         - IMG/                (Contains all images generated)
# By using ./image_angle_to_h5.py, data from Directory_name/driving_log.csv & Directory_name/IMG are combined
# correspondingly and stored in a h5 file. (If the dataset in the h5 file already exists, new records will auto append
# to that dataset.)


import matplotlib.image as mpimg
import numpy as np
import cv2
from time import time
import csv
import h5py
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def read_csv(csv_path, img_dir):
    """ Read from the csv file, return image paths and their corresponding angles

    :param str csv_path: Where is the input csv file
    :param str img_dir: Where do images locate (images directory)

    :return: (img_path, angle)
        img_path:  (list) image paths get from csv file
        angle: (list) corresponding angles from images from img_path
    :rtype: tuple(list, list)

    """
    img_path = []
    angle = []
    with open(csv_path, newline='') as f:
        log_reader = csv.reader(f, delimiter=',')
        for row in log_reader:
            center_img = row[0]
            loc = img_dir + center_img.split('/')[-1]
            img_path.append(loc)
            angle.append(row[3])
    angle = np.array(angle, dtype=np.float32)
    print(angle.dtype)
    print('Is image path len same as angles? ', len(img_path) == angle.shape[0])
    return img_path, angle


# pre-process input images,
def preprocess_img(img_path, height, width, channel):
    """ pre-process images located in img_path parameter
        by resizing them to desired shape then horizontally flip them to double the data size

    :param list of str img_path: locations of all images in the list
    :param int height: desired height
    :param int width: desired width
    :param int channel: desired color channel
    :return: images with desired shape as a numpy array
    :rtype: Numpy array
    """
    start = time()
    loop_time = time()
    img_stack = np.zeros((1, height, width, channel))
    for i, loc in enumerate(img_path):
        img = mpimg.imread(loc)
        img = img[54:]
        img = cv2.resize(img, (width, height))
        img_flip = cv2.flip(img, 1)
        img = np.expand_dims(img, axis=0)
        img_flip = np.expand_dims(img_flip, axis=0)
        img_stack = np.concatenate((img_stack, img, img_flip))

        del img, img_flip
        if i % 500 == 0:
            abc = time()
            print(i, ' read and resize takes: ', abc - start, '500 diff is: ',abc - loop_time)
            loop_time = abc
    print('Whole process takes that long: ', time() - start)
    img_stack = img_stack[1:]
    return img_stack


def save_to_h5(data_dict, h5_file):
    f = h5py.File(h5_file)
    for k, v in data_dict.items():
        if k in f:  # if dataset already exists
            dset = f[k]
            ori_nb = f[k].shape[0]
            added_nb = v.shape[0]
            nb_after_resize = ori_nb + added_nb
            dset.resize(nb_after_resize, axis=0)
            dset[-added_nb:] = v
        else:
            max_shape = [None,]
            for i in v.shape[1:]:
                max_shape.append(i)
            f.create_dataset(k, data=v, maxshape=max_shape)
    f.flush()
    f.close()


def save_preprocess_img(path, storage_path):
    data_dict = {}
    csv_path = path + 'driving_log.csv'
    img_dir = path + 'IMG/'
    img_paths, angles = read_csv(csv_path, img_dir)
    img_stack = preprocess_img(img_paths, height=66, width=200, channel=3)

    y = np.zeros(1)
    for val in angles:
        a1 = val
        a2 = - a1
        y = np.append(y, (a1, a2))
    y = y[1:]
    img_stack = img_stack.astype(np.int16)

    for i in range(5):
        img_stack, y = shuffle(img_stack, y, random_state=i)
    X_train, X_val, y_train, y_val = train_test_split(img_stack, y, test_size=0.2, random_state=43)

    print('img_stack shape is: ', img_stack.shape)
    print('angles list here: ', len(y))
    data_dict = {'X_train': X_train, 'X_val': X_val, 'y_train': y_train, 'y_val': y_val}
    save_to_h5(data_dict, storage_path)
    del data_dict, y, img_stack, angles, X_train, X_val, y_train, y_val


dir_index = ['bridge_2', 'last_sandy']
base_path = '../data/'
h5_path = '../data/h5_storage.h5'
path_list = [base_path + name + '/' for name in dir_index]
for path in path_list:
    save_preprocess_img(path, h5_path)
