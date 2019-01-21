import random
import numpy as np
from scipy.ndimage import gaussian_filter, zoom, rotate, map_coordinates

def random_crop3d(input_, target_):
    zoom_rates = [1.2, 1.3, 1.4]
    zoom_rate  = random.choice(zoom_rates)


    zoom_input  = zoom(input_,  zoom_rate)
    zoom_target = zoom(target_, zoom_rate)

    zoom_shape, img_shape = zoom_input.shape, input_.shape
    dx = random.randint(0, zoom_shape[0] - img_shape[0])
    dy = random.randint(0, zoom_shape[1] - img_shape[1])

    zoom_input  = zoom_input[dx:dx + img_shape[0], dy:dy + img_shape[1]]
    zoom_target = zoom_target[dx:dx + img_shape[0], dy:dy + img_shape[1]]
    return zoom_input, zoom_target

def random_flip(input_, target_):
    flip = random.randint(0, 7) # 0, 1, 2

    if   flip == 0:
        return input_[:, ::-1, ::-1],    target_[:, ::-1, ::-1]
    elif flip == 1:
        return input_[:, ::-1, :],    target_[:, ::-1, :]
    elif flip == 2:
        return input_[:, :, ::-1], target_[:, :, ::-1]
    elif flip == 3:
        return input_, target_
    elif flip == 4:
        return input_[::-1, ::-1, ::-1], target_[::-1, ::-1, ::-1]
    elif flip == 5:
        return input_[::-1, ::-1, :], target_[::-1, ::-1, :]
    elif flip == 6:
        return input_[::-1, :, ::-1], target_[::-1, :, ::-1]
    elif flip == 7:
        return input_[::-1, :, :], target_[::-1, :, :]

def random_rotate(input_, target_):
    param_list = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    angle=random.choice(param_list)

    rotate_input  = rotate(input_, angle,
                           reshape=False)
    rotate_target = rotate(target_, angle,
                           reshape=False)
    return rotate_input, rotate_target


def elastic_transform(input_, target_, weight_=None, param_list=None, random_state=None):
    if param_list is None:
        ##HHE
        #param_list = [(0, 1),(2, 1), (5, 2), (4, 3)]

        ##HE
        param_list = [(1,0.5)]
        #param_list = [(0, 1),(0, 1),(0, 1), (1, 1), (1.5, 2), (1, 0.5)]

    alpha, sigma = random.choice(param_list)

    assert len(input_.shape)==3
    shape = input_.shape

    if random_state is None:
       random_state = np.random.RandomState(None)    

    dx = gaussian_filter((random_state.rand(*shape[0:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape[0:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    #print(np.mean(dx), np.std(dx), np.min(dx), np.max(dx))

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    transformed = []
    for image in [input_, target_]:
        new = np.zeros(shape)
        if len(shape) == 3:
            for i in range(image.shape[2]):
                new[:, :, i] = map_coordinates(image[:, :, i], indices, order=1, mode="reflect").reshape(shape[0:2])
        else:
            new[:, :] = map_coordinates(image[:, :], indices, order=1, mode="reflect").reshape(shape)
        transformed.append(new)

    return transformed


ARG_TO_DICT = {
        "crop":random_crop3d,
        "flip":random_flip,
        "elastic":elastic_transform,
        "rotate":random_rotate
        }

def get_preprocess(preprocess_list):
    if not preprocess_list:
        return []
    return [ARG_TO_DICT[p] for p in preprocess_list.split(",")]

