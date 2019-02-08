# coding=utf-8
import numpy as np
import os
import tarfile
import zipfile
import urllib

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def try_download_and_extract(name="mnist"):
    directory = "/Users/jwl1993/dataset/"
    data_url = None
    if name == "mnist":
        data_name = "mnist/"
    elif name == "cifar10":
        data_name = "cifar10/"
        data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    elif name == "cifar100":
        data_name = "cifar100/"
        data_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

    data_directory = directory

    if not os.path.exists(data_directory + data_name) and name != "mnist" :
        print("There is no data set of %s, so we try to download" %name)
        os.makedirs(data_directory)

        compressed_file = data_url.split("/")[-1]
        location = os.path.join(data_directory, compressed_file)
        download_file, _ = urllib.urlretrieve(url=data_url, filename=location)
        print("Download done")

        if download_file.endswith(".zip") :
            zipfile.ZipFile(file=download_file, mode="r").extractall(directory)
        elif download_file.endswith(".tar.gz") :
            tarfile.open(name=download_file, mode="r:gz").extractall(directory)

        print("Extract done")

        os.remove(location)

def get_dataset(name="mnist", mode="train", data_dir="/Users/jwl1993/dataset/") :
    if name == "mnist" :
        directory = data_dir + name + "/"
        dataset = input_data.read_data_sets(directory)
    elif name == "cifar10" :
        dataset = None
        #[XXX] Processing data for cifar 10
        pass
    elif name == "cifar100" :
        dataset = None
        #[XXX] Processing data for cifar 100
        pass

    return dataset
