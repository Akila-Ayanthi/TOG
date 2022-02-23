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

fpath = '/home/dissana8/LAB/Visor/'
savepath = '/home/dissana8/TOG/Adv_images'

for path, subdirs, files in os.walk(fpath):
    for name in files:
        print(os.path.join(path, name))
        print(name)
        # input_img = Image.open(os.path.join(path, name))
        # x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
        # x_adv_untargeted = tog_untargeted(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
        # x_adv_untargeted = x_adv_untargeted.save(os.path.join(savepath, )))