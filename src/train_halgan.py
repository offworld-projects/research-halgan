from collections import defaultdict
import pickle
from PIL import Image
import imageio
import math
from scipy.misc import imresize
from scipy.stats import gamma, gaussian_kde
import os
import numpy as np
import tensorflow as tf

import keras.backend as K
from keras.layers.core import Lambda
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Multiply, Dropout, LeakyReLU, Embedding, UpSampling2D, Activation, BatchNormalization, Add, Concatenate
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
from keras.layers.merge import _Merge
from functools import partial

def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """Calculates the gradient penalty loss for a batch of "averaged" samples."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = GRADIENT_PENALTY_WEIGHT * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. """

    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def build_generator(nb_classes, latent_size):
    '''we will map a pair of (z, L), where z is a latent vector and L is a
    label drawn from P_c, to image space (..., 1, 64, 64)'''

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    rel_pos = Input(shape=(nb_classes,), dtype='float32')
    e1 = Dense(latent_size)(rel_pos)

    # merge them!
    merged = Multiply()([latent, e1])

    d1 = Reshape((1, 1, 128))(merged)

    # upsample to (2,2)
    u0_1 = UpSampling2D()(d1)
    conv0_1 = Conv2D(64, 4,
            strides=1,
            padding='same')(u0_1)
    conv0_1 = BatchNormalization()(conv0_1)
    conv0_1 = LeakyReLU()(conv0_1)

    # upsample to (4,4)
    u0 = UpSampling2D()(conv0_1)
    conv0 = Conv2D(64, 4,
            strides=1,
            padding='same')(u0)
    conv0 = BatchNormalization()(conv0)
    conv0 = LeakyReLU()(conv0)

    # upsample to (8, 8)
    u1 = UpSampling2D()(conv0)
    conv1 = Conv2D(64, 4,
            strides=1,
            padding='same')(u1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)

    # upsample to (16, 16)
    u2 = UpSampling2D()(conv1)
    conv2 = Conv2D(32, 4,
            strides=1,
            padding='same')(u2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)

    # upsample to (32, 32)
    u3 = UpSampling2D()(conv2)
    conv3 = Conv2D(32, 4,
            strides=1,
            padding='same')(u3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)

    # upsample to (64, 64)
    u4 = UpSampling2D()(conv3)
    conv4 = Conv2D(16, 4,
            strides=1,
            padding='same')(u4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU()(conv4)

    # # couple of more layers of convolution
    conv5 = Conv2D(8, 4,
            strides=1,
            padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU()(conv5)

    fake_image = Conv2D(3, 4,
            strides=1,
            activation='tanh',
            padding='same')(conv5)

    return Model(inputs=[latent, rel_pos], outputs=fake_image)

def build_discriminator():
    '''build a relatively standard conv net, with LeakyReLUs as suggested in
    the reference paper'''
    image = Input(shape=(64, 64, 3))
    conv0 = Conv2D(32, 4,
                strides=1,
                name='conv5x5_0',
                padding='same')(image)
    conv0 = LeakyReLU()(conv0)

    conv1 = Conv2D(32, 4,
                strides=2,
                name='conv5x5_1',
                padding='same')(conv0)
    conv1 = LeakyReLU()(conv1)

    conv2 = Conv2D(32, 4,
                strides=2,
                padding='same',
                name='conv5x5_2')(conv1)
    conv2 = LeakyReLU()(conv2)

    conv3 = Conv2D(64, 4,
                strides=2,
                name='conv5x5_3',
                padding='same')(conv2)
    conv3 = LeakyReLU()(conv3)

    conv4 = Conv2D(64, 4,
                strides=2,
                name='conv5x5_4',
                padding='same')(conv3)
    conv4 = LeakyReLU()(conv4)

    conv5 = Conv2D(64, 4,
                strides=2,
                name='conv5x5_5',
                padding='same')(conv4)
    conv5 = LeakyReLU()(conv5)

    conv6 = Conv2D(128, 4,
                strides=2,
                name='conv5x5_6',
                padding='same')(conv5)
    conv6 = LeakyReLU()(conv6)
    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to. In this case, that is orientation to goal.
    inter = Reshape((128,))(conv6)
    fake = Dense(1, activation='linear', name='generation')(inter)
    aux = Dense(nb_classes, activation='linear', name='auxiliary')(inter)

    return Model(inputs=image, outputs=[fake, aux])

class TrainIterator():
    def __init__(self, NAME, batchsize):
        self.labels = np.load(os.path.join(base_dir, 'relpos.npy'))
        imgPaths = np.load(os.path.join(base_dir, 'filepaths.npy'))
        # sample 2000 random images
        arr = np.arange(imgPaths.shape[0])
        np.random.shuffle(arr)
        imgPaths = imgPaths[arr[:2000]]
        self.labels = self.labels[arr[:2000]]
        self.trainImgs = []
        for path in imgPaths:
            self.trainImgs.append(preprocess_img(imageio.imread(path)))
        self.batchsize = batchsize
        # generate gaussian non-parameterics for label distribution
        self.label_distribution = gaussian_kde(np.transpose(self.labels))

    def __len__(self):
        return len(self.trainImgs)

    def next(self):
        '''sample a batch'''
        # sample indices
        idxs = np.random.randint(0, len(self.trainImgs), size=self.batchsize)
        image_batch = np.array([self.trainImgs[i] for i in idxs])
        label_batch = self.labels[idxs]
        # add noise
        label_batch[:,0] += np.random.normal(scale=0.01, size=(self.batchsize,))
        label_batch[:,0] = np.abs(label_batch[:,0])
        label_batch[:,1] += np.random.normal(scale=0.02, size=(self.batchsize,))
        return image_batch, label_batch

class FailIterator():
    def __init__(self, NAME, batchsize):
        imgPaths = os.listdir('{}/fail/fail/'.format(base_dir))
        imgPaths = [os.path.join('{}/fail/fail/'.format(base_dir), path) for path in imgPaths]
        imgPaths = np.array(imgPaths)
        # sample 10000 random images
        arr = np.arange(imgPaths.shape[0])
        np.random.shuffle(arr)
        imgPaths = imgPaths[arr[:10000]]
        self.trainImgs = []
        for path in imgPaths:
            self.trainImgs.append(preprocess_img(imageio.imread(path)))
        self.batchsize = batchsize

    def __len__(self):
        return len(self.trainImgs)

    def next(self):
        '''sample a batch'''
        # sample indices
        idxs = np.random.randint(0, len(self.trainImgs), size=self.batchsize)
        image_batch = np.array([self.trainImgs[i] for i in idxs])
        return image_batch


def preprocess_img(img):
    '''resize and convert to [-1,1]'''
    img = imresize(img, (64, 64))
    return 2*(img/255.) - 1

if __name__ == '__main__':

    NAME = 'MiniWorld-SimToReal1-v0'
    ROOT = '.'

    GRADIENT_PENALTY_WEIGHT = 10

    K.set_image_dim_ordering('tf')
    # optional - set up the gpu
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # set_session(tf.Session(config=config))

    # batch and latent size taken from the paper
    nb_epochs = 50000
    batch_size = 64
    latent_size = 128
    nb_classes = 2

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0001
    adam_beta_1 = 0.5
    adam_beta_2 = 0.9

    # build the discriminator
    discriminator = build_discriminator()

    # build the model to penalize gradients
    input_real = Input(shape=(64, 64, 3))
    input_fake = Input(shape=(64, 64, 3))
    avg_input = RandomWeightedAverage()([input_real, input_fake])
    real_out, aux_real = discriminator(input_real)
    fake_out, aux_fake = discriminator(input_fake)
    avg_out, _ = discriminator(avg_input)
    # The gradient penalty loss function requires the input averaged samples to get gradients. However,
    # Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
    # of the function with the averaged samples here.
    partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=avg_input)

    # now the real/fake outputs are penalized as normal
    # but average out is penalized for gradients
    discriminator_grad_penalty = Model(
        inputs=[input_real, input_fake],
        outputs=[real_out, aux_real, fake_out, aux_fake, avg_out])
    # compile with a relative weighting on losses
    discriminator_grad_penalty.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=[wasserstein, 'mse',\
              wasserstein, 'mse',\
              partial_gp_loss],
        loss_weights=[1.0, 10.0,
                      1.0, 10.0,
                      1.0]
    )

    # build the generator
    generator = build_generator(nb_classes, latent_size)
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                      loss='binary_crossentropy')

    fail_image = Input(shape=(64, 64, 3))
    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(nb_classes,), dtype='float32')

    # get a fake image
    diff = generator([latent, image_class])
    # add back the input to diff
    fake = Add()([diff, fail_image])
    # renormalize to [-1,1]
    fake = Activation('tanh')(fake)

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fool, aux = discriminator(fake)
    combined = Model(inputs=[fail_image, latent, image_class], outputs=[diff, fool, aux])

    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['mse', wasserstein, 'mse'],
        loss_weights=[1.0, 1.0, 10.0]
    )

    # in wgans, reals are 1 and fakes are -1
    y_real = np.array([1.] * batch_size)
    y_fake = np.array([-1.] * batch_size)
    dummy_y = np.array([0.] * batch_size)

    print("Generator Model")
    print(generator.summary())
    print("Discriminator Model")
    print(discriminator.summary())
    input("Press enter to continue")

    # load data and rescale to range [-1, 1]
    datagen = ImageDataGenerator(preprocessing_function=preprocess_img)
    base_dir = '../data/{}-regression-1000/'.format(NAME)

    trainIterator = TrainIterator(NAME, batch_size)
    failIterator = FailIterator(NAME, batch_size)
    nb_batches = len(trainIterator)/batch_size
    d_iters = 5
    nb_iters = int((nb_epochs * nb_batches)/(d_iters + 1))

    train_history = defaultdict(list)
    import datetime
    datestamp = datetime.datetime.now().strftime('%Y-%m-%d|%H:%M:%S')
    logdir = os.path.join(ROOT, 'experiments',
        'halgan-{}'.format(NAME), datestamp)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, 'checkpoints'))
    os.makedirs(os.path.join(logdir, 'generated-goals'))

    epoch = 0
    print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
    progress_bar = Progbar(target=nb_iters)
    epoch_gen_loss = []
    epoch_disc_loss = []

    for it in range(nb_iters):
        progress_bar.update(it)
        # train disc. first, more than generator
        for d_it in range(d_iters):
            # get a batch of real images
            image_batch, label_batch = trainIterator.next()
            fail_batch = failIterator.next()
            # generate a new batch of noise
            noise = np.random.normal(1., .1, (batch_size, latent_size))
            sampled_labels = np.transpose(trainIterator.label_distribution.resample(batch_size))
            generated_images = generator.predict(
                [noise, sampled_labels], verbose=0)
            # now these generated images are only differences. So we add them
            # back to the input fail images to get game screens.
            generated_images += fail_batch
            generated_images = np.tanh(generated_images)

            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator_grad_penalty.train_on_batch(
                [image_batch, generated_images],
                [y_real, label_batch, y_fake, sampled_labels, dummy_y]))

        fail_batch = failIterator.next()
        noise = np.random.normal(1., .1, (batch_size, latent_size))
        sampled_labels = np.transpose(trainIterator.label_distribution.resample(batch_size))

        # we want to train the generator to trick the discriminator
        # For the generator, we want all the {fake, not-fake} labels to say
        # not-fake
        trick = np.ones(batch_size)

        # train combined model
        epoch_gen_loss.append(combined.train_on_batch(
            [fail_batch, noise, sampled_labels], [np.zeros_like(fail_batch), trick, sampled_labels]))

        if it % int(nb_batches/(d_iters+1)) == 0:
            generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
            discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

            # generate an epoch report on performance
            train_history['generator'].append(generator_train_loss)
            train_history['discriminator'].append(discriminator_train_loss)
            pickle.dump({'train': train_history,},
                open(os.path.join(logdir, 'train-history.pkl'), 'wb'))
            if epoch % 10== 0:
                # save weights every epoch 100 epochs
                if epoch%10==0:
                    generator.save_weights(os.path.join(logdir, 'checkpoints',
                        'params_generator_epoch_{0:03d}.hdf5'.format(epoch)), True)
                    discriminator_grad_penalty.save_weights(os.path.join(logdir, 'checkpoints',
                        'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch)), True)

                fail_batch = failIterator.next()
                # generate some images to display
                fail_batch = fail_batch[:10]
                noise = np.random.normal(1., .1, (10, latent_size))
                sampled_labels = np.transpose(trainIterator.label_distribution.resample(10))
                generated_images = generator.predict(
                    [noise, sampled_labels], verbose=0)

                # add the diff images back into fail_batch
                generated_images += fail_batch
                generated_images = np.tanh(generated_images)

                # arrange them into a grid
                img = ((np.concatenate(generated_images, axis=0) + 1)*127.5).astype(np.uint8)
                # concatenate the fail images to the left of this
                fail_batch = ((fail_batch + 1)*127.5).astype(np.uint8)
                img = np.concatenate([np.concatenate(fail_batch, axis=0), img], axis=1)

                imageio.imsave(os.path.join(logdir, 'generated-goals',
                    'plot_epoch_{0:03d}_generated.png'.format(epoch)), img)
                np.savetxt(os.path.join(logdir, 'generated-goals',
                    'plot_epoch_{0:03d}_labels.txt'.format(epoch)), sampled_labels)

            # Epoch over!
            epoch += 1
            print('\nEpoch {} of {}'.format(epoch + 1, nb_epochs))
            epoch_gen_loss = []
            epoch_disc_loss = []
