from dataset_utils.preprocessing import letterbox_image_padded
from misc_utils.visualization import visualize_detections
from keras import backend as K
from models.ssd import SSD512
from PIL import Image
from tog.attacks import *
import os
K.clear_session()


weights = './model_weights/SSD512.h5'  # TODO: Change this path to the victim model's weights

detector = SSD512(weights=weights)


eps = 8 / 255.       # Hyperparameter: epsilon in L-inf norm
eps_iter = 2 / 255.  # Hyperparameter: attack learning rate
n_iter = 10          # Hyperparameter: number of attack iterations

fpath = '~/LAB/Visor/'

for path, subdirs, files in os.walk(fpath):
    for name in files:
        print(os.path.join(path, name))