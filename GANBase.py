import sys
import time
import random
import math
import copy
from ops import *
from utils import *
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
from tensorflow.contrib.opt import MovingAverageOptimizer

class GANBase(object):

    def __init__(self, sess, args):
        self.sess = sess
        self.dataset_name = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.checkpoint = args.checkpoint
        if args.phase=="service":
            self.request_dir = args.request_dir

        self.epoch = args.epoch
        self.iterations_per_epoch = args.iteration
        self.batch_size = args.batch_size
        self.virtual_batches = args.virtual_batches
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.histogram_freq = args.histogram_freq
        self.keep_checkpoints = args.keep_checkpoints
        if self.keep_checkpoints==0:
            self.keep_checkpoints = None

        self.img_size = args.img_size
        self.random_flip = args.random_flip
        self.save_morphs = args.save_morphs

        self.bn_options = {}
        self.bn_options["type"] = args.bn_type
        self.bn_options["momentum"] = args.bn_momentum
        if self.bn_options["type"]=='batch_renorm':
            self.bn_options["renorm_clipping"] = {}
            self.bn_options["renorm_clipping"]["rmax"] = args.bn_renorm_rmax
            self.bn_options["renorm_clipping"]["dmax"] = args.bn_renorm_dmax
            self.bn_options["renorm_momentum"] = args.bn_renorm_momentum
            self.bn_options["shared_renorm"] = args.bn_renorm_shared

        self.conv_options = {}
        self.conv_options["padding_type"] = args.conv_padding
        self.conv_options["sn"] = args.sn