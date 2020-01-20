import sys
import time
import random
from ops import *
from utils import *
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
from tensorflow.contrib.opt import MovingAverageOptimizer

class BigGAN(object):

    def __init__(self, sess, args):
        self.model_name = "BigGAN"  # name for checkpoint
        self.sess = sess
        self.dataset_name = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.checkpoint = args.checkpoint
        if args.phase == "service":
            self.request_dir = args.request_dir

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.batch_size = args.batch_size
        self.virtual_batches = args.virtual_batches
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.keep_checkpoints = args.keep_checkpoints
        if self.keep_checkpoints==0:
            self.keep_checkpoints=None

        self.img_size = args.img_size
        self.random_flip = args.random_flip
        self.depth = args.img_size.bit_length()-2
        self.save_morphs = args.save_morphs
        self.n_labels = args.n_labels
        self.acgan = self.n_labels>0
        self.cls_loss_type = args.cls_loss_type
        self.label_file = args.label_file
        self.weight_file = args.weight_file
        self.g_first_level_dense_layer = args.g_first_level_dense_layer
        self.g_other_level_dense_layer = args.g_other_level_dense_layer
        self.d_cls_dense_layers = args.d_cls_dense_layers
        self.g_final_layer = args.g_final_layer
        self.g_final_layer_shortcuts = args.g_final_layer_shortcuts
        self.g_final_layer_shortcuts_after = args.g_final_layer_shortcuts_after
        self.g_final_mixed_conv = args.g_final_mixed_conv
        self.g_final_mixed_conv_stacks = args.g_final_mixed_conv_stacks
        self.g_final_mixed_conv_mix_kernel = args.g_final_mixed_conv_mix_kernel
        if self.g_final_layer:
            self.depth += 1

        self.activation = args.activation
        if self.activation=='relu':
            self.activation_fn = relu
        elif self.activation=='prelu':
            self.activation_fn = prelu
        elif self.activation=='lrelu':
            def lrelu_p(alpha):
                def lrelu_a(x):
                    return lrelu(x,alpha)
                return lrelu_a
            self.activation_fn = lrelu_p(0.2)
        else:
            raise ValueError("Unknown activation function: "+str(self.activation))


        if self.g_final_mixed_conv and not self.g_final_layer:
            raise ValueError("g_final_mixed_conv set, but not g_final_layer")

        if self.acgan:
            self.d_cls_loss_weight = args.d_cls_loss_weight
            self.g_cls_loss_weight = args.g_cls_loss_weight
            self.save_cls_samples = args.save_cls_samples
            self.cls_loss_weight_file = args.cls_loss_weights
            if self.cls_loss_weight_file:
                self.cls_loss_weights = list(map(float, open(self.cls_loss_weight_file).read().split()))
                if len(self.cls_loss_weights) != self.n_labels:
                    raise ValueError('Number of class loss weights does not match n_labels ('+str(len(self.cls_loss_weights))+" vs. "+str(self.n_labels)+")")
            else: self.cls_loss_weights = [1.0] * self.n_labels

        """ Generator """
        self.ch = args.ch
        self.upsampling_method = args.upsampling_method
        self.g_grow_factor = args.g_grow_factor
        self.g_regularization_method = args.g_regularization
        self.g_regularization_factor = args.g_regularization_factor

        self.z_dim = args.z_dim  # dimension of noise-vector
        self.first_split_ratio = args.first_split_ratio
        if self.z_dim%self.depth!=0 and self.first_split_ratio==1:
            self.z_dim=self.z_dim + self.depth - self.z_dim%self.depth
            print("Warning: z_dim must be divisible by ",self.depth,", changing to ",self.z_dim)
        self.z_reconstruct = args.z_reconstruct

        self.gan_type = args.gan_type
        self.d_loss_func = args.d_loss_func if args.d_loss_func else self.gan_type

        if self.d_loss_func.__contains__('wgan'):
            self.gradient_penalty_type = self.gan_type
        elif self.d_loss_func.__contains__('dragan'):
            self.gradient_penalty_type = 'dragan'
        else:
            self.gradient_penalty_type = None


        self.use_gradient_penalty = bool(self.gradient_penalty_type)


        if self.use_gradient_penalty and self.z_reconstruct:
            raise ValueError('Z reconstruction is currently not supported when gradient penalty is used (WGAN and DRAGAN)')

        """ Discriminator """
        self.n_critic = args.n_critic
        self.sn = args.sn
        self.ld = args.ld
        self.bn_in_d = args.bn_in_d
        self.d_ch = args.d_ch
        if self.d_ch <= 0: self.d_ch = self.ch
        self.d_grow_factor = args.d_grow_factor

        self.sample_num = args.sample_num  # number of generated images to be saved
        self.static_sample_z = args.static_sample_z
        self.static_sample_seed = args.static_sample_seed
        self.z_trunc_train = args.z_trunc_train
        self.z_trunc_sample = args.z_trunc_sample
        self.test_num = args.test_num
        self.sample_ema = args.sample_ema

        if self.sample_ema not in ['ema', 'noema', 'both']:
            raise ValueError('Invalid mode for sample_ema specified')

        self.generate_noema_samples = self.sample_ema!='ema'
        self.generate_ema_samples = self.sample_ema!='noema'

        # train
        self.g_learning_rate = args.g_lr
        self.d_learning_rate = args.d_lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.moving_decay = args.moving_decay

        self.custom_dataset = True
        self.c_dim = args.c_dim
        self.alpha_mask = args.alpha_mask
        self.data, self.labels = load_data(dataset_name=self.dataset_name, label_file=self.label_file, weight_file=self.weight_file)

        self.dataset_num = len(self.data)

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        print()

        print("##### Information #####")
        print("# BigGAN", self.img_size)
        print("# gan type:", self.gan_type)
        print("# dataset:", self.dataset_name)
        print("# dataset number:", self.dataset_num)
        print("# batch_size:", self.batch_size)
        print("# epoch:", self.epoch)
        print("# iteration per epoch:", self.iteration)

        print()

        print("##### Generator #####")
        print("# spectral normalization:", self.sn)
        print("# learning rate:", self.g_learning_rate)

        print()

        print("##### Discriminator #####")
        print("# discriminator updates per generator update:", self.n_critic)
        print("# spectral normalization:", self.sn)
        print("# learning rate:", self.d_learning_rate)
        if self.gan_type != self.d_loss_func:
            print("# discriminator loss:", self.d_loss_func)

    ##################################################################################
    # Generator
    ##################################################################################

    def round_up(self, val, multiple):
        return (int(val) + multiple - 1) // multiple * multiple

    def scale_channels(self, ch, factor):
        return self.round_up(int(ch * factor),8)

    def g_channels_for_block(self, b_i, b_count):
        return self.scale_channels(self.ch,self.g_grow_factor**(b_count-b_i-1))


    def generator(self, z, cls_z, is_training=True, reuse=False, custom_getter=None):

        opt = {"sn": self.sn,
               "is_training": is_training,
               "upsampling_method": self.upsampling_method,
               "act": self.activation_fn}

        if is_training:
            if self.g_regularization_method=='none':
                opt["conv_regularizer"]=None
                opt["fc_regularizer"]=None
            elif self.g_regularization_method=='ortho':
                opt["conv_regularizer"]=orthogonal_regularizer(self.g_regularization_factor,type='ortho')
                opt["fc_regularizer"]=orthogonal_regularizer_fc(self.g_regularization_factor,type='ortho')
            elif self.g_regularization_method=='ortho_cosine':
                opt["conv_regularizer"]=orthogonal_regularizer(self.g_regularization_factor,type='ortho_cosine')
                opt["fc_regularizer"]=orthogonal_regularizer_fc(self.g_regularization_factor,type='ortho_cosine')
            elif self.g_regularization_method=='l2':
                opt["conv_regularizer"]=tf.contrib.layers.l2_regularizer(self.g_regularization_factor)
                opt["fc_regularizer"]=tf.contrib.layers.l2_regularizer(self.g_regularization_factor)
            else:
                raise ValueError("Unknown regularization method: "+str(self.g_regularization_method))
        else:
            opt["conv_regularizer"]=None
            opt["fc_regularizer"]=None

        with tf.variable_scope("generator", reuse=reuse, custom_getter=custom_getter):
            if self.first_split_ratio>1:
                split_dim = self.z_dim // (self.depth - 1 + self.first_split_ratio)
                first_split_dim = self.z_dim - (self.depth - 1) * split_dim
            else:
                split_dim = self.z_dim // self.depth
                first_split_dim = split_dim

            z_split = tf.split(z, num_or_size_splits=[first_split_dim] + ([split_dim] * (self.depth - 1)), axis=-1)

            if self.acgan:
                scls_z = tf.reshape(cls_z, shape=[-1, 1, 1, self.n_labels])
                for i in range(len(z_split)):
                    z_split[i]=tf.concat([z_split[i],scls_z],axis=-1)

            if   self.img_size==64: block_info={"counts": [1, 1, 1, 1], "sa_index": 3}
            elif self.img_size==128: block_info={"counts": [1, 1, 1, 1, 1], "sa_index": 4}
            elif self.img_size==256: block_info={"counts": [1, 2, 1, 1, 1], "sa_index": 4}
            elif self.img_size==512: block_info={"counts": [1, 2, 1, 1, 2], "sa_index": 3}
            else: raise ValueError("Invalid image size specified: "+str(self.img_size))

            ch_mul = 2**(len(block_info["counts"])-1)
            ch = self.g_channels_for_block(0, len(block_info["counts"]))

            if self.g_first_level_dense_layer:
                f_width = self.round_up((first_split_dim+self.n_labels)*1.85,8)
                if opt["act"]==relu:
                    # omit scope for backward compatibility
                    x=fully_connected(z_split[0], units=f_width, scope='dense1', opt=opt)
                    x=relu(x)
                    x=fully_connected(x, units=4*4*ch, scope='dense2', opt=opt)
                else:
                    with tf.variable_scope('first'):
                        x=fully_connected(z_split[0], units=f_width, scope='dense1', opt=opt)
                        x=opt["act"](x)
                        x=fully_connected(x, units=4 * 4 * ch, scope='dense2', opt=opt)
            else: x=fully_connected(z_split[0], units=4*4*ch, scope='dense', opt=opt)

            x=tf.reshape(x, shape=[-1, 4, 4, ch])

            z_i = 1
            b_i = 0

            for block_count in block_info["counts"]:
                scope='resblock_up_'+str(ch_mul)
                for sb_i in range(block_count):
                    if block_count>1: scope=scope+'_'+str(sb_i)
                    if self.g_other_level_dense_layer:
                        with tf.variable_scope('z'+str(ch_mul)):
                            f_width = self.round_up((split_dim+self.n_labels)*1.25,8)
                            block_z = fully_connected(z_split[z_i], units=f_width, scope='dense1', opt=opt)
                            block_z = opt["act"](block_z)
                    else:
                        block_z = z_split[z_i]
                    x=resblock_up_condition(x, block_z, channels=ch, use_bias=False, opt=opt, scope=scope)
                    z_i+=1

                b_i+=1
                if b_i==block_info["sa_index"]:
                    x=self_attention_2(x, channels=ch, opt=opt, scope='self_attention')

                ch = self.g_channels_for_block(b_i, len(block_info["counts"]))
                ch_mul=ch_mul//2

            x = batch_norm(x, opt=opt)
            x = opt["act"](x)

            if self.g_final_layer:
                with tf.variable_scope('final'):

                    if self.g_final_layer_shortcuts:
                        lsa_i = self.g_final_layer_shortcuts_after
                        slice_units = self.round_up(self.ch/(8.0-lsa_i*1.5),4)
                        final_slice_units = self.round_up(slice_units*(2.5-lsa_i*0.25),4)
                        final_channels = slice_units*(self.g_final_mixed_conv_stacks-lsa_i) + final_slice_units
                        slices = []
                        if lsa_i==0:
                            slices.append(conv(x, channels=slice_units, kernel=3, stride=1, pad=1, use_bias=False, opt=opt, scope="slice1"))
                    else:
                        final_channels = self.scale_channels(self.ch, 0.5)

                    use_bias = True

                    if use_bias:
                        zfi_ch = final_channels * 2
                    else:
                        zfi_ch = split_dim * 2
                    final = fully_connected(z_split[self.depth - 1], units=zfi_ch, scope='dense', opt=opt)
                    final = opt["act"](final)
                    final_scale = fully_connected(final, units=final_channels, scope='dense2', opt=opt)
                    final_scale = tf.reshape(final_scale, shape=[-1, 1, 1, final_channels])

                    if use_bias:
                        final_bias = fully_connected(final, units=final_channels, scope='dense_bias', opt=opt)
                        final_bias = tf.reshape(final_bias, shape=[-1, 1, 1, final_channels])

                    if self.g_final_mixed_conv:
                        for i in range(0,self.g_final_mixed_conv_stacks):
                            x = clown_conv(x, self.ch, scope="clown"+('' if i==0 else str(i+1)), opt=opt)
                            if self.g_final_layer_shortcuts and (i+1)>=lsa_i:
                                s_units = final_slice_units if (i==self.g_final_mixed_conv_stacks-1) else slice_units
                                slices.append(conv(x, channels=s_units, kernel=3, stride=1, pad=1, use_bias=False, opt=opt, scope='slice'+str(i+2)))

                    if self.g_final_layer_shortcuts:
                        x = tf.concat(slices, axis=-1)
                    else:
                        x = conv(x, channels=final_channels, kernel=3, stride=1, pad=1, use_bias=False, opt=opt)


                    if use_bias:
                        x = x * final_scale + final_bias
                    else:
                        x = x * final_scale
                    x = prelu(x)
                x = conv(x, channels=self.c_dim, kernel=self.g_final_mixed_conv_mix_kernel, stride=1, pad=(self.g_final_mixed_conv_mix_kernel-1)//2, use_bias=False, opt=opt, scope='G_logit')

            else:
                x = conv(x, channels=self.c_dim, kernel=3, stride=1, pad=1, use_bias=False, opt=opt, scope='G_logit')

            x = tanh(x)

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def d_channels_for_block(self, b_i):
        return self.scale_channels(self.d_ch,self.d_grow_factor**b_i)

    def discriminator(self, x, is_training=True, reuse=False):

        opt = {"sn": self.sn,
               "is_training": is_training,
               "bn_in_d": self.bn_in_d,
               "act": self.activation_fn}

        with tf.variable_scope("discriminator", reuse=reuse):
            ch = self.d_channels_for_block(0)

            if   self.img_size==64: block_info={"counts": [1, 1, 1, 1], "sa_index": 1}
            elif self.img_size==128: block_info={"counts": [1, 1, 1, 1, 1], "sa_index": 1}
            elif self.img_size==256: block_info={"counts": [1, 1, 1, 2, 1], "sa_index": 2}
            elif self.img_size==512: block_info={"counts": [1, 2, 1, 1, 2], "sa_index": 2}
            else: raise ValueError("Invalid image size specified: "+str(self.img_size))

            if self.c_dim==4 and self.alpha_mask:
                rgb, alpha = tf.split(x, num_or_size_splits=[3,1], axis=-1)
                rgb_filtered = ((rgb + 1.0) * (alpha + 1.0) * 0.5) - 1.0
                x = tf.concat([rgb_filtered, alpha], axis=-1)


            b_i=0
            ch_mul=1
            for block_count in block_info["counts"]:
                scope='resblock_down_'+str(ch_mul)
                for sb_i in range(block_count):
                    if block_count>1: scope=scope+'_'+str(sb_i)
                    x=resblock_down(x, channels=ch, use_bias=False, opt=opt, scope=scope)

                b_i+=1
                if b_i==block_info["sa_index"]:
                    x=self_attention_2(x, channels=ch, opt=opt, scope='self_attention')

                ch=self.d_channels_for_block(b_i)
                ch_mul=ch_mul*2

            ch=self.d_channels_for_block(b_i-1)  # last layer has same width as previous one

            x = resblock(x, channels=ch, use_bias=False, opt=opt, scope='resblock')
            x = opt["act"](x)

            features = global_sum_pooling(x)


            x = fully_connected(features, units=1, opt=opt, scope='D_logit')

            outputs = {"real": x}

            if self.acgan:
                opt["sn"]=False
                if self.d_cls_dense_layers:
                    with tf.variable_scope("classification", reuse=reuse):
                        cls_funits1 = self.round_up(ch/16.0+self.n_labels*1.25,8)
                        y = fully_connected(features, units=cls_funits1, opt=opt, scope='dense1')
                        y = opt["act"](y)
                        cls_funits2 = self.round_up(cls_funits1/4.0+self.n_labels*1.1,4)
                        y = fully_connected(y, units=cls_funits2, opt=opt, scope='dense2')
                        y = opt["act"](y)
                        y = fully_connected(y, units=self.n_labels, opt=opt, scope='DC_logit')
                else:
                    y = fully_connected(features, units=self.n_labels, opt=opt, scope='DC_logit')
                outputs["cls"] = y

            if self.z_reconstruct:
                opt["sn"] = False
                y = fully_connected(features, units=self.z_dim, opt=opt, scope='z_reconstruct')

                branch = fully_connected(features, units=self.scale_channels(self.z_dim,1.5), opt=opt, scope='z_reconstruct_res1')
                branch = batch_norm(branch, opt)
                branch = relu(branch)
                branch = fully_connected(branch, units=self.z_dim, opt=opt, scope='z_reconstruct_res2')

                y = y + branch

                outputs["z"] = y

            return outputs

    def gradient_penalty(self, real, fake):
        if self.gradient_penalty_type == 'dragan':
            eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1., dtype=gan_dtype)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

            fake = real + 0.5 * x_std * eps

        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1., dtype=gan_dtype)
        interpolated = real + alpha * (fake - real)

        logit = self.discriminator(interpolated, reuse=True)["real"]

        grad = tf.gradients(logit, interpolated)[0]  # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=1)  # l2 norm

        GP = 0

        # WGAN - LP
        if self.gradient_penalty_type == 'wgan-lp':
            GP = self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif self.gradient_penalty_type == 'wgan-gp' or self.gradient_penalty_type == 'dragan':
            GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        # images
        Image_Data_Class = ImageData(self.img_size, self.c_dim, self.custom_dataset, self.random_flip)
        if self.acgan:
            inputs = tf.data.Dataset.from_tensor_slices((self.data,self.labels))
        else: inputs = tf.data.Dataset.from_tensor_slices(self.data)

        gpu_device = '/gpu:0'
        inputs = inputs.\
            apply(shuffle_and_repeat(self.dataset_num)).\
            apply(map_and_batch(Image_Data_Class.image_processing_with_labels if self.acgan else Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).\
            apply(prefetch_to_device(gpu_device, self.batch_size))

        inputs_iterator = inputs.make_one_shot_iterator()

        if self.acgan:
            self.inputs, self.label_input = inputs_iterator.get_next()
        else:
            self.inputs = inputs_iterator.get_next()

        # noises
        if self.z_trunc_train:
            self.z = tf.random.truncated_normal(shape=[self.batch_size, 1, 1, self.z_dim], name='random_z', dtype=gan_dtype)
        else:
            self.z = tf.random.normal(shape=[self.batch_size, 1, 1, self.z_dim], name='random_z', dtype=gan_dtype)

        if self.z_trunc_sample==self.z_trunc_train:
            self.test_z = self.z
        else:
            if self.z_trunc_sample:
                self.test_z = tf.random.truncated_normal(shape=[self.batch_size, 1, 1, self.z_dim], name='test_random_z', dtype=gan_dtype)
            else:
                self.test_z = tf.random.normal(shape=[self.batch_size, 1, 1, self.z_dim], name='test_random_z', dtype=gan_dtype)

        """ Loss Function """
        # output of D for real images

        d_outputs_real = self.discriminator(self.inputs)
        real_logits = d_outputs_real["real"]

        if self.acgan:
            real_cls_logits = d_outputs_real["cls"]
            self.zero_cls_z = tf.zeros(shape=(self.batch_size,self.n_labels), dtype=gan_dtype)
            self.cls_z = tf.placeholder(gan_dtype, [self.batch_size, self.n_labels], name='cls_z')
        else:
            self.zero_cls_z = None
            self.cls_z = None

        # output of D for fake images
        fake_images = self.generator(self.z,self.cls_z)
        d_outputs_fake = self.discriminator(fake_images, reuse=True)
        fake_logits = d_outputs_fake["real"]

        if self.acgan:
            fake_cls_logits = d_outputs_fake["cls"]

        if self.z_reconstruct:
            self.reconstructed_z = tf.reshape(d_outputs_fake["z"], shape=[-1, 1, 1, self.z_dim])

        if self.use_gradient_penalty:
            GP = self.gradient_penalty(real=self.inputs, fake=fake_images)
        else:
            GP = 0

        self.d_classification_loss = 0
        self.z_reconstruct_loss = 0

        if self.acgan:
            cls_weights = tf.constant(self.cls_loss_weights, dtype=gan_dtype)

            def cls_loss(truth,answer):
                if self.cls_loss_type == 'logistic':
                    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=truth,logits=answer)*cls_weights)
                elif self.cls_loss_type == 'euclidean':
                    return tf.norm((answer - truth) * cls_weights, ord='euclidean')
                else:
                    raise ValueError("Invalid label loss type: "+self.cls_loss_type)

            self.d_classification_loss = self.d_cls_loss_weight * cls_loss(self.label_input,real_cls_logits)

        if self.z_reconstruct:
            loss_f = tf.constant(1.0/self.z_dim)
            self.z_reconstruct_loss = tf.norm(self.z - self.reconstructed_z, ord='euclidean') * loss_f

        # get loss for discriminator
        self.d_loss = (discriminator_loss(self.d_loss_func, real=real_logits, fake=fake_logits)) + GP + self.d_classification_loss + (self.z_reconstruct_loss*5.0)

        self.g_classification_loss = 0

        if self.acgan:
            self.g_classification_loss = self.g_cls_loss_weight * cls_loss(self.cls_z,fake_cls_logits)

        # get loss for generator
        self.g_loss = generator_loss(self.gan_type, fake=fake_logits, real=real_logits) + (self.g_classification_loss) + (self.z_reconstruct_loss*5.0)
        if self.g_regularization_method!='none':
            self.g_loss = tf.add_n([self.g_loss] + tf.losses.get_regularization_losses())

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        # optimizers
        if self.virtual_batches > 1:

            self.d_opt = tf.train.AdamOptimizer(self.d_learning_rate, beta1=self.beta1, beta2=self.beta2)

            self.opt = MovingAverageOptimizer(
                tf.train.AdamOptimizer(self.g_learning_rate, beta1=self.beta1, beta2=self.beta2),
                average_decay=self.moving_decay)
            ema = self.opt._ema

            def virtual_batch_steps(opt, loss, vars):
                grad_factor = tf.constant(1.0/self.virtual_batches, dtype=gan_dtype)
                acc_vars = [tf.Variable(tf.zeros_like(v.read_value()), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]) for v in vars]
                zero_ops = [v.assign(tf.zeros_like(v)) for v in acc_vars]
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    grads = opt.compute_gradients(loss, var_list=vars)
                acc_ops = [acc_vars[i].assign_add(gv[0]) for i, gv in enumerate(grads)]
                step = opt.apply_gradients([(acc_vars[i]*grad_factor, gv[1]) for i, gv in enumerate(grads)])
                return zero_ops, acc_ops, step

            self.g_zero_step, self.g_acc_steps, self.g_train_step = virtual_batch_steps(self.opt, self.g_loss, g_vars)
            self.d_zero_step, self.d_acc_steps, self.d_train_step = virtual_batch_steps(self.d_opt, self.d_loss, d_vars)

        else:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.d_optim = tf.train.AdamOptimizer(self.d_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_vars)
                self.opt = MovingAverageOptimizer(tf.train.AdamOptimizer(self.g_learning_rate, beta1=self.beta1, beta2=self.beta2), average_decay=self.moving_decay)
                ema = self.opt._ema
                self.g_optim = self.opt.minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test

        def ema_getter(getter, name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            ema_var = ema.average(var) if ema else None
            return ema_var if ema_var else var

        self.fake_images = self.generator(self.test_z, self.zero_cls_z, is_training=False, reuse=True, custom_getter=ema_getter)

        if self.static_sample_z or self.save_morphs:
            self.sample_z = tf.placeholder(gan_dtype, [self.batch_size, 1, 1, self.z_dim], name='sample_z')
            self.z_generator = self.generator(self.sample_z, self.cls_z, reuse=True, is_training=False, custom_getter=ema_getter)

        if self.static_sample_z:
            rounded_n = self.round_up(self.sample_num, self.batch_size)
            if self.z_trunc_sample:
                self.sample_z_val = self.sess.run(tf.random.truncated_normal(shape=[rounded_n, 1, 1, self.z_dim], name='sample_z_gen', seed=self.static_sample_seed))
            else:
                self.sample_z_val = self.sess.run(tf.random.normal(shape=[rounded_n, 1, 1, self.z_dim], name='sample_z_gen', seed=self.static_sample_seed))

            self.sample_fake_images = self.z_generator

            if self.generate_noema_samples:
                self.sample_fake_images_noema = self.generator(self.sample_z, self.cls_z, reuse=True, is_training=False)

            if self.acgan:
                np.random.seed(self.static_sample_seed)
                self.sample_cls_z = self.draw_n_tags(rounded_n)

        else:
            self.sample_fake_images = self.fake_images

            if self.generate_noema_samples:
                self.sample_fake_images_noema = self.generator(self.test_z, self.zero_cls_z, is_training=False, reuse=True)



        """ Summary """
        self.d_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.dc_sum = tf.summary.scalar("d_cls_loss", self.d_classification_loss)
        self.gc_sum = tf.summary.scalar("g_cls_loss", self.g_classification_loss)

    ##################################################################################
    # Train
    ##################################################################################

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = self.opt.swapping_saver(max_to_keep=self.keep_checkpoints)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter

            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        past_g_loss = -1.
        past_g_cls_loss = -1.
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, self.iteration):

                # update D network
                if self.acgan:

                    if self.virtual_batches > 1:

                        self.sess.run(self.d_zero_step,feed_dict=self.rnd_cls_feed_dict())

                        d_loss = 0
                        d_cls_loss = 0
                        for i in range(self.virtual_batches):
                            summary_str, s2_str, loss, cls_loss, *_ = self.sess.run([self.d_sum, self.dc_sum, self.d_loss, self.d_classification_loss] + self.d_acc_steps,feed_dict=self.rnd_cls_feed_dict())
                            d_loss += loss
                            d_cls_loss += cls_loss

                        self.sess.run(self.d_train_step)

                        d_loss /= self.virtual_batches
                        d_cls_loss /= self.virtual_batches

                    else:
                        _, summary_str, s2_str, d_loss, d_cls_loss = self.sess.run([self.d_optim, self.d_sum, self.dc_sum, self.d_loss, self.d_classification_loss],feed_dict=self.rnd_cls_feed_dict())

                    self.writer.add_summary(summary_str, counter)
                    self.writer.add_summary(s2_str, counter)
                else:
                    if self.virtual_batches>1:

                        self.sess.run(self.d_zero_step)

                        d_loss = 0
                        for i in range(self.virtual_batches):
                            summary_str, loss, *_ = self.sess.run([self.d_sum, self.d_loss] + self.d_acc_steps)
                            d_loss += loss

                        self.sess.run(self.d_train_step)

                        d_loss /= self.virtual_batches

                    else:
                        _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss])

                    self.writer.add_summary(summary_str, counter)

                # update G network
                g_loss = None
                g_cls_loss = None
                if (counter - 1) % self.n_critic == 0:
                    if self.acgan:
                        if self.virtual_batches > 1:

                            self.sess.run(self.g_zero_step,feed_dict=self.rnd_cls_feed_dict())

                            g_loss = 0
                            g_cls_loss = 0
                            for i in range(self.virtual_batches):
                                summary_str, s2_str, loss, cls_loss, *_ = self.sess.run([self.g_sum, self.gc_sum, self.g_loss, self.g_classification_loss] + self.g_acc_steps,feed_dict=self.rnd_cls_feed_dict())
                                g_loss += loss
                                g_cls_loss += cls_loss

                            self.sess.run(self.g_train_step)

                            g_loss /= self.virtual_batches
                            g_cls_loss /= self.virtual_batches
                        else:
                            _, summary_str, s2_str, g_loss, g_cls_loss = self.sess.run([self.g_optim, self.g_sum, self.gc_sum, self.g_loss, self.g_classification_loss],feed_dict=self.rnd_cls_feed_dict())

                        self.writer.add_summary(summary_str, counter)
                        self.writer.add_summary(s2_str, counter)
                        past_g_loss = g_loss
                        past_g_cls_loss = g_cls_loss
                    else:
                        if self.virtual_batches > 1:

                            self.sess.run(self.g_zero_step)

                            g_loss = 0
                            for i in range(self.virtual_batches):
                                summary_str, loss, *_ = self.sess.run([self.g_sum, self.g_loss]+self.g_acc_steps)
                                g_loss += loss

                            self.sess.run(self.g_train_step)

                            g_loss /= self.virtual_batches
                        else:
                            _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss])

                        self.writer.add_summary(summary_str, counter)
                        past_g_loss = g_loss

                # display training status
                counter += 1
                if g_loss == None:
                    g_loss = past_g_loss
                    if self.acgan: g_cls_loss = past_g_cls_loss

                if self.acgan:
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f, d_loss: %.8f, d_cls_loss: %.8f x%.1f, g_loss: %.8f, g_cls_loss: %.8f x%.1f" \
                          % (epoch, idx, self.iteration, time.time() - start_time, d_loss, d_cls_loss, self.d_cls_loss_weight, g_loss, g_cls_loss, self.g_cls_loss_weight))
                else:
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                                  % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                sys.stdout.flush()

                # save training results every X steps
                if np.mod(idx + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

                # generate samples every X steps
                if np.mod(idx + 1, self.print_freq) == 0:
                    manifold_h = int(np.floor(np.sqrt(self.sample_num)))
                    manifold_w = int(np.floor(np.sqrt(self.sample_num)))
                    tot_num_samples = manifold_w * manifold_h

                    batches = int(np.ceil(tot_num_samples/self.batch_size))

                    def generate_with(gen):
                        for b in range(batches):
                            if self.static_sample_z:

                                feed_dict = {
                                    self.sample_z: self.sample_z_val[b * self.batch_size:(b + 1) * self.batch_size],
                                }

                                if self.acgan:
                                    feed_dict[self.cls_z] = self.sample_cls_z[b * self.batch_size:(b + 1) * self.batch_size]

                                batch = self.sess.run(gen, feed_dict=feed_dict)
                            else:
                                batch = self.sess.run(gen)

                            if b==0: samples = batch
                            else: samples = np.append(samples, batch, axis=0)

                        return samples

                    if self.generate_ema_samples:
                        save_images(generate_with(self.sample_fake_images)[:manifold_h * manifold_w, :, :, :],
                                    [manifold_h, manifold_w],
                                    './' + self.sample_dir + '/' + self.model_name + '_ema_{:02d}_{:05d}.png'.format(
                                        epoch, idx + 1))

                    if self.generate_noema_samples:
                        save_images(generate_with(self.sample_fake_images_noema)[:manifold_h * manifold_w, :, :, :],
                                    [manifold_h, manifold_w],
                                    './' + self.sample_dir + '/' + self.model_name + '_noema_{:02d}_{:05d}.png'.format(
                                        epoch, idx + 1))

                    if self.save_morphs:

                        def select():
                            ri=np.random.randint(self.sample_num)
                            if self.acgan:
                                return self.sample_z_val[ri], np.array(list(map(float,self.sample_cls_z[ri])))
                            else:
                                return self.sample_z_val[ri], None

                        z_a, cz_a = select()
                        z_b, cz_b = select()
                        z_c, cz_c = select()
                        z_d, cz_d = select()

                        morph_padding = 1
                        inter_z_flat = []
                        inter_cz_flat = [] if self.acgan else None
                        for x in range(-morph_padding, manifold_w + morph_padding):
                            rx = x / (manifold_w - 1)
                            for y in range(-morph_padding, manifold_h + morph_padding):
                                ry = y / (manifold_h - 1)
                                inter_z_flat.append(z_a * (1 - rx) * (1 - ry) + z_b * (rx) * (1 - ry) + z_c * (1 - rx) * (ry) + z_d * (rx) * (ry))

                                if self.acgan:
                                    inter_cz_flat.append(cz_a * (1 - rx) * (1 - ry) + cz_b * (rx) * (1 - ry) + cz_c * (1 - rx) * (ry) + cz_d * (rx) * (ry))

                        samples = self.generate(inter_z_flat, inter_cz_flat)

                        save_images(samples[:(manifold_h+morph_padding*2) * (manifold_w+morph_padding*2), :, :, :],
                                    [manifold_h+morph_padding*2, manifold_w+morph_padding*2],
                                    './' + self.sample_dir + '/' + self.model_name + '_morph_{:02d}_{:05d}.png'.format(
                                        epoch, idx + 1))

                    if self.acgan and self.save_cls_samples:
                        def select_by_tag(tag_index):
                            x=0
                            while True:
                                ri=np.random.randint(len(self.labels))
                                if x>=1000:
                                    print("Warning: did not find any samples for tag index",tag_index,", picking at random")
                                if self.labels[ri][tag_index]>0 or x>=1000:
                                    return self.labels[ri]
                                x+=1

                        np.random.seed(epoch * self.iteration + idx)
                        rti = np.random.randint(self.n_labels)

                        z_flat = []
                        cz_flat = []
                        i = 0
                        for x in range(manifold_w):
                            for y in range(manifold_h):
                                z_flat.append(self.sample_z_val[i])
                                cz_flat.append(select_by_tag(rti))
                                i+=1

                        samples = self.generate(z_flat, cz_flat)

                        save_images(samples[:(manifold_h) * (manifold_w), :, :, :],
                                    [manifold_h, manifold_w],
                                    './' + self.sample_dir + '/' + self.model_name + '_cls_{:02d}_{:05d}_{:03d}.png'.format(
                                        epoch, idx + 1, rti))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            # self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        if self.sn :
            sn = '_sn'
        else :
            sn = ''

        return "{}_{}_{}_{}_{}{}".format(
            self.model_name, self.dataset_name, self.gan_type, self.img_size, self.z_dim, sn)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)


    def load_checkpoint(self, checkpoint_path):
        ckpt_name = os.path.basename(checkpoint_path)
        self.saver.restore(self.sess, checkpoint_path)
        counter = int(ckpt_name.split('-')[-1])
        print(" [*] Successfully read {}".format(ckpt_name))
        return True, counter


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if self.checkpoint:
            return self.load_checkpoint(self.checkpoint)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            return self.load_checkpoint(ckpt.model_checkpoint_path)
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        samples = self.sess.run(self.fake_images)

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    self.sample_dir + '/' + self.model_name + '_epoch%02d' % epoch + '_visualize.png')


    def service(self):
        tf.global_variables_initializer().run()

        #self.saver = tf.train.Saver()
        self.saver = self.opt.swapping_saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")


        while True:
            files = [os.path.join(self.request_dir, f) for f in os.listdir(self.request_dir) if
                     os.path.isfile(os.path.join(self.request_dir, f)) and f.lower().endswith('.txt')]

            for f in files:
                cmds, z_samples = read_vectors(f)

                use_ema=True

                if "quit" in cmds:
                    os.remove(f)
                    return

                if "ema" in cmds:
                    use_ema=cmds["ema"]=='1'

                if "load_checkpoint" in cmds:
                    self.load_checkpoint(cmds["load_checkpoint"])

                cls_z = None

                if self.acgan:
                    cls_z = []
                    for i, z in enumerate(z_samples):
                        cls_z.append(z[self.z_dim:])
                        z_samples[i] = z[:self.z_dim]

                for i, z in enumerate(z_samples):
                    z_samples[i] = [[z]]

                if len(z_samples)>0:
                    samples = self.generate(z_samples, cls_z, ema=use_ema)
                    out_path = os.path.join(self.request_dir, os.path.splitext(os.path.basename(f))[0] + ".png")
                    save_images(samples,[1, len(samples)],out_path.replace(".png",".tmp.png"))
                    os.rename(out_path.replace(".png",".tmp.png"),out_path)
                os.remove(f)

            time.sleep(0.005)



    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        for i in range(self.test_num):
            samples = self.sess.run(self.fake_images)

            save_images(samples[:image_frame_dim * image_frame_dim, :, :, :],
                        [image_frame_dim, image_frame_dim],
                        result_dir + '/' + self.model_name + '_test_{}.png'.format(i))


    def generate(self, zv, cls_zv, ema=True):

        zvc = zv.copy()
        while len(zvc) % self.batch_size != 0:
            zvc.append(zvc[0])
            if cls_zv: cls_zv.append(cls_zv[0])

        batches = int(np.ceil(len(zvc) / self.batch_size))
        for b in range(batches):
            feed_dict = {self.sample_z: zvc[b * self.batch_size:(b + 1) * self.batch_size]}
            if cls_zv:
                feed_dict[self.cls_z]=cls_zv[b * self.batch_size:(b + 1) * self.batch_size]

            generator = self.sample_fake_images if ema else self.sample_fake_images_noema

            batch = self.sess.run(generator, feed_dict)
            if b == 0:
                samples = batch
            else:
                samples = np.append(samples, batch, axis=0)

        if len(zvc)==len(zv): return samples
        else: return samples[:len(zv)]

    def draw_n_tags(self, n):
        tags=[]
        for i in range(n):
            ri=np.random.randint(len(self.labels))
            tags.append(self.labels[ri])
        return tags

    def rnd_cls_feed_dict(self):
        return {self.cls_z: self.draw_n_tags(self.batch_size)}
