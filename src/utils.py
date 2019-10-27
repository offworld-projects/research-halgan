import numpy as np
from scipy.misc import imresize
import math

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Conv2D, BatchNormalization, Permute, Reshape, Lambda, LeakyReLU, MaxPooling2D
from keras import backend as K

from rl.processors import Processor, SqueezeProcessor

class MiniWorldProcessor(Processor):
    def __init__(self):
        self.last_health = 1000000

    def process_observation(self, observation):
        # resize to 64x64
        observation[0] = imresize(observation[0], (64, 64))
        # [0, 255] -> [0., 1.]
        observation[0] = observation[0].astype(np.float32)
        observation[0] = observation[0]/255.

        # convert direction to [0,2pi] counterclock from x axis
        # currently it is -inf to +inf clockwise from x axis
        yaw = abs(observation[1][2])%(2*math.pi)
        if observation[1][2] > 0:
            yaw = 2*math.pi - yaw
        observation[1][2] = yaw

        observation[1] = np.array(observation[1])
        # a=input()
        # if a =='s':
            # import matplotlib.pyplot as plt
            # plt.imshow(observation[0])
            # plt.show()
            # plt.close()
        return observation

    def process_state_batch(self, batch):
        '''
        Only needed if state is comprised of images and configs!
        keras-rl batches are sampled as list of
        [(img1, config1), (img2, config2), ...]. This converts to model input
        expected format which is [(batch_imgs), (batch_configs)]. Takes into
        account history.
        '''
        imgs_batch = []
        configs_batch = []
        for exp in batch:
            imgs = []
            configs = []
            for state in exp:
                imgs.append(np.expand_dims(state[0], 0))
                configs.append(np.expand_dims(np.expand_dims(state[1], 0), 0))
            imgs_batch.append(np.concatenate(imgs, -1))
            configs_batch.append(np.concatenate(configs, 1))
        imgs_batch = np.concatenate(imgs_batch, 0)
        configs_batch = np.concatenate(configs_batch, 0)
        return (imgs_batch, configs_batch)


def simple_architecture(nb_actions):
    inchannels = 3

    image_input = Input(
	shape=(64, 64, inchannels),
	dtype='float32',
	name='agent/image_input')

    conv_1 = Conv2D(4, (5,5),
	padding='same',
	name='agent/conv5x5_1',
        kernel_initializer='glorot_uniform')(image_input)
    conv_1 = MaxPooling2D(padding='same')(conv_1)
    conv_1 = LeakyReLU()(conv_1)

    conv_2 = Conv2D(8, (5,5),
	padding='same',
	name='agent/conv5x5_2')(conv_1)
    conv_2 = MaxPooling2D()(conv_2)
    conv_2 = LeakyReLU()(conv_2)

    conv_3 = Conv2D(16, (5,5),
	padding='same',
	name='agent/conv5x5_3')(conv_2)
    conv_3 = MaxPooling2D()(conv_3)
    conv_3 = LeakyReLU()(conv_3)

    conv_4 = Conv2D(32, (5,5),
	padding='same',
	name='agent/conv5x5_4')(conv_3)
    conv_4 = MaxPooling2D()(conv_4)
    conv_4 = LeakyReLU()(conv_4)

    flattened = Flatten()(conv_4)

    fc_image = Dense(32, name='agent/image_fc')(flattened)
    fc_image = LeakyReLU()(fc_image)

    q = Dense(nb_actions, activation='linear', name='agent/q')(fc_image)
    model = Model(inputs=image_input, outputs=q)

    print(model.summary())
    return model
