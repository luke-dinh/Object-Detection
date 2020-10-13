import os
import re
import zipfile

from tensorflow import keras

#Link to get COCO 2017 Dataset
#Dataset is optional. In this code I choose COCO 2017 dataset. 

url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
filename = os.path.join(os.getcwd(), "data.zip")
keras.utils.get_file(filename, url)


with zipfile.ZipFile("data.zip", "r") as z_fp:
    z_fp.extractall("./")

