import os, sys, argparse, json
import numpy as np
import tensorflow as tf
import gym
import gym_miniworld

import keras
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session

from rl.agents import HALGANDDPGAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory, HALGANMemory
from rl.processors import Processor
from rl.callbacks import ModelIntervalCheckpoint, FileLogger, MemoryIntervalCheckpoint, TrajectoryDump
from rl.random import GaussianWhiteNoiseProcess

from utils import *

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general arguments
    parser.add_argument('--gpu_id', help='GPU id to use', type=int, default=0)
    parser.add_argument('--seed', help='RNG seed', type=int, default=123)
    parser.add_argument('--save_every', help='save network weights every x steps', type=int, default=1e4)
    parser.add_argument('--log_every', help='update logs every x episodes', type=int, default=10)
    parser.add_argument('--weights_file', help='load network weights from checkpoint file', default=None)
    parser.add_argument('--memory_dir', help='load agent memory from this directory', default=None)
    parser.add_argument('--mode', help='experimental condition', default='halgan', choices=['halgan', 'vanilla', 'rig-', 'vae-her', 'her'])
    parser.add_argument('--enjoy', help='run a pretrained policy', type=bool, default=False)

    # environment arguments
    parser.add_argument('--env', help='environment ID', default='MiniWorldSimToReal1-v0')
    parser.add_argument('--step_penalty', help='step penalty', type=float, default=0.)
    parser.add_argument('--power_penalty_mult', help='scaling of penalty on power', type=float, default=0.)
    parser.add_argument('--out_of_bounds_penalty', help='penalty for exceeding bounding box', type=float, default=0)

    # algorithmic arguments
    parser.add_argument('--replay_warmup', help='states in the replay before training starts', type=int, default=1000)
    parser.add_argument('--replay_capacity', help='max length of replay buffer', type=int, default=100000)
    parser.add_argument('--max_steps', help='maximum number of steps train agent', type=int, default=10000000)
    parser.add_argument('--max_episode_steps', help='maximum number of steps in an episode', type=int, default=200)
    parser.add_argument('--max_dist_to_goal', help='oracle/GAN hallucinate only this far back', type=int, default=16)
    parser.add_argument('--gamma', help='discount factor', type=float, default=0.99)
    parser.add_argument('--eps_start', help='starting random exploration', type=float, default=1.)
    parser.add_argument('--eps_end', help='minimum random exploration', type=float, default=0.05)
    parser.add_argument('--eps_end_step', help='step at which minimum exploration is hit', type=float, default=100000)
    parser.add_argument('--hallucination_start', help='starting percentatge of hallucinations', type=float, default=20.)
    parser.add_argument('--hallucination_end_step', help='steps to decay epsilon to is final value', type=int, default=200000)
    args = parser.parse_args()

    # set up the environment (optional)
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # set_session(tf.Session(config=config))

    ROOT = '.'
    gym.undo_logger_setup()


    # Create the environment and extract the number of actions.
    env = gym.make(args.env)
    env.human_collects = False
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    # set random seeds
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    env.seed(args.seed)

    # Next, we build a very simple actor model.
    image_input = Input(
	shape=(64, 64, 3),
	dtype='float32',
	name='actor/image_input')

    conv_1 = Conv2D(4, (5,5),
	padding='same',
	name='actor/conv5x5_1',
        kernel_initializer='glorot_normal')(image_input)
    conv_1 = MaxPooling2D(padding='same')(conv_1)
    conv_1 = LeakyReLU()(conv_1)

    conv_2 = Conv2D(8, (5,5),
	padding='same',
	name='actor/conv5x5_2',
        kernel_initializer='glorot_normal')(conv_1)
    conv_2 = MaxPooling2D()(conv_2)
    conv_2 = LeakyReLU()(conv_2)

    conv_3 = Conv2D(16, (5,5),
	padding='same',
	name='actor/conv5x5_3',
        kernel_initializer='glorot_normal')(conv_2)
    conv_3 = MaxPooling2D()(conv_3)
    conv_3 = LeakyReLU()(conv_3)

    conv_4 = Conv2D(32, (5,5),
	padding='same',
	name='actor/conv5x5_4',
        kernel_initializer='glorot_normal')(conv_3)
    conv_4 = MaxPooling2D()(conv_4)
    conv_4 = LeakyReLU()(conv_4)

    flattened = Flatten()(conv_4)

    fc_image = Dense(32, name='actor/image_fc', kernel_initializer='glorot_normal')(flattened)
    fc_image = LeakyReLU()(fc_image)
    a = Dense(2, activation='tanh', name='actor/a', kernel_initializer='glorot_normal')(fc_image)
    actor = Model(inputs=image_input, outputs=a)
    print(actor.summary())

    # Next, we build a very simple critic model.
    v_conv_1 = Conv2D(4, (5,5),
	padding='same',
	name='critic/conv5x5_1',
        kernel_initializer='glorot_normal')(image_input)
    v_conv_1 = MaxPooling2D(padding='same')(v_conv_1)
    v_conv_1 = LeakyReLU()(v_conv_1)

    v_conv_2 = Conv2D(8, (5,5),
	padding='same',
	name='critic/conv5x5_2',
        kernel_initializer='glorot_normal')(v_conv_1)
    v_conv_2 = MaxPooling2D()(v_conv_2)
    v_conv_2 = LeakyReLU()(v_conv_2)

    v_conv_3 = Conv2D(16, (5,5),
	padding='same',
	name='critic/conv5x5_3',
        kernel_initializer='glorot_normal')(v_conv_2)
    v_conv_3 = MaxPooling2D()(v_conv_3)
    v_conv_3 = LeakyReLU()(v_conv_3)

    v_conv_4 = Conv2D(32, (5,5),
	padding='same',
	name='critic/conv5x5_4',
        kernel_initializer='glorot_normal')(v_conv_3)
    v_conv_4 = MaxPooling2D()(v_conv_4)
    v_conv_4 = LeakyReLU()(v_conv_4)

    v_flattened = Flatten()(v_conv_4)
    action_input = Input(shape=(nb_actions,), name='action_input')
    v_concat =  Concatenate()([v_flattened, action_input])

    fc = Dense(32, name='critic/fc', kernel_initializer='glorot_normal')(v_concat)
    fc = LeakyReLU()(fc)
    v = Dense(1, activation='linear', name='critic/v', kernel_initializer='glorot_normal')(fc)
    critic = Model(inputs=[image_input, action_input], outputs=v)
    print(critic.summary())

    # setup agent
    if args.env == 'MiniWorld-SimToReal1Cont-v0':
        processor =  MiniWorldProcessor()
    elif args.env == 'MiniWorld-SimToReal2Cont-v0':
        processor =  MiniWorldProcessor()
    else:
        raise NotImplementedError
    memory = HALGANMemory(limit=args.replay_capacity, window_length=1)
    random_process = GaussianWhiteNoiseProcess(size=nb_actions, mu=0., sigma=0.1)
    halgana = HALGANDDPGAgent(
        nb_actions=nb_actions, actor=actor, critic=critic,
        critic_action_input=action_input, memory=memory,
        nb_steps_warmup_critic=args.replay_warmup,
        nb_steps_warmup_actor=args.replay_warmup, random_process=random_process,
        gamma=args.gamma, target_model_update=1e-3, processor=processor)
    halgana.compile([Adam(lr=1e-5), Adam(lr=1e-4)], metrics=['mae'])

    # load agent weights if pretrained
    if args.weights_file:
        print("Loading pretrained weights from: {}".format(args.weights_file))
        halgana.load_weights(args.weights_file)

    # load and configure the GAN
    def get_percent_hallucination(step):
        # linear annealing after warmup period
         if step < args.hallucination_end_step:
             return args.hallucination_start*\
                 (1-float(step-args.replay_warmup)/(args.hallucination_end_step-args.replay_warmup))
         else:
             return 0.
    halgana.mode = args.mode
    if args.mode == 'vanilla':
        print("Vanilla dqn")
        halgana.percent_hallucination = lambda x: 0.
    if args.mode == 'halgan':
        from halgan import build_generator
        gan = build_generator(nb_classes=2, latent_size=128)
        if args.env == 'MiniWorld-SimToReal1Cont-v0':
            ganpath = os.path.join(ROOT,'data',args.env, 'halgan.hdf5')
        else:
            raise NotImplementedError
        print("Loading GAN weights from: {}".format(ganpath))
        halgana.configure_gan(gan, 128, ganpath)
        halgana.percent_hallucination = get_percent_hallucination
    if args.mode == 'rig-':
        from beta_vae import build_vae, preprocess_img
        import imageio
        encoder, _, vae, _, _ = build_vae()
        vaepath = os.path.join(ROOT, '..',
            'data', args.env, 'params_vae.hdf5')
        print("Loading VAE weights from: {}".format(vaepath))
        vae.load_weights(vaepath)
        halgana.encoder = encoder
        halgana.percent_hallucination = get_percent_hallucination
        # load random near-goal images
        halgana.near_goal = []
        base_dir = '../data/{}/'.format(NAME)
        imgPaths = np.load(os.path.join(base_dir, 'filepaths.npy'))
        if args.env == 'MiniWorld-SimToReal1Cont-v0':
            idxs = np.load(os.path.join(base_dir, 'randomized_idxs_success.npy'))
            imgPaths = imgPaths[idxs[:2000]]
            labels = np.load(os.path.join(base_dir, 'relpos.npy'))[idxs[:2000]]
            for i in range(imgPaths.shape[0]):
                if labels[i][0] < 0.01 and labels[i][1] < 0.1:
                    halgana.near_goal.append(preprocess_img(imageio.imread(imgPaths[i])))
        else:
            NotImplementedError
    if args.mode == 'vae-her':
        from beta_vae import build_vae, preprocess_img
        import imageio
        encoder, _, vae, _, _ = build_vae()
        vaepath = os.path.join(ROOT, '..', 'data', args.env, 'params_vae.hdf5')
        print("Loading VAE weights from: {}".format(vaepath))
        vae.load_weights(vaepath)
        halgana.encoder = encoder
        halgana.percent_hallucination = get_percent_hallucination
    if args.mode == 'her':
        print("Running naive HER")
        halgana.percent_hallucination = get_percent_hallucination
    else:
        raise NotImplementedError

    # final few configuration
    # how far ahead to sample hallucinations
    halgana.max_dist_to_goal = args.max_dist_to_goal
    # environment
    halgana.ENV_NAME = args.env
    halgana.action_box = [env.action_space.low, env.action_space.high]
    # exploration
    def get_epsilon(step):
        # linear annealing after warmup period
        if step < args.replay_warmup:
            return args.eps_start
        elif step > args.eps_end_step:
            return args.eps_end
        else:
            return args.eps_start - (step-args.replay_warmup)*(args.eps_start-args.eps_end)/(args.eps_end_step-args.replay_warmup)
    halgana.eps = get_epsilon

    # create logging and checkpoints
    import datetime
    datestamp = datetime.datetime.now().strftime('%Y-%m-%d|%H:%M:%S')
    log_dir = '{}/experiments/halgan-{}/halgan-ddpg/{}'.format(ROOT, args.env, datestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # save experiment config
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))
    # experiment logger
    logfile = os.path.join(log_dir, 'training.log')
    checkpoint_dir = log_dir + '/checkpoints/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_checkpoint = os.path.join(checkpoint_dir, 'weights_{step}.h5f')
    callbacks = [FileLogger(logfile, interval=args.log_every),]

    # learn! (or visualize!)
    if args.enjoy:
        halgana.test(env,
            step=1000000,
            nb_episodes=2000,
            visualize=True,
            nb_max_episode_steps=args.max_episode_steps,
            callbacks=callbacks)
    else:
        callbacks += [ModelIntervalCheckpoint(filepath=model_checkpoint, interval=args.save_every),]
        halgana.fit(env,
            callbacks=callbacks,
            nb_steps=args.max_steps,
            nb_max_episode_steps=args.max_episode_steps,
            visualize=False,
            verbose=1,
            log_interval=1000)

if __name__ == '__main__':
    main()
