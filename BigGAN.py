import sys
import time
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

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.img_size = args.img_size
        self.depth = args.img_size.bit_length()-2

        """ Generator """
        self.ch = args.ch

        self.z_dim = args.z_dim  # dimension of noise-vector
        if self.z_dim%self.depth!=0:
            self.z_dim=self.z_dim + self.depth - self.z_dim%self.depth
            print("Warning: z_dim must be divisible by ",self.depth,", changing to ",self.z_dim)

        self.gan_type = args.gan_type

        """ Discriminator """
        self.n_critic = args.n_critic
        self.sn = args.sn
        self.ld = args.ld

        self.sample_num = args.sample_num  # number of generated images to be saved
        self.test_num = args.test_num

        # train
        self.g_learning_rate = args.g_lr
        self.d_learning_rate = args.d_lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.moving_decay = args.moving_decay

        self.custom_dataset = False

        if self.dataset_name == 'mnist':
            self.c_dim = 1
            self.data = load_mnist()

        elif self.dataset_name == 'cifar10':
            self.c_dim = 3
            self.data = load_cifar10()

        else:
            self.c_dim = 3
            self.data = load_data(dataset_name=self.dataset_name)
            self.custom_dataset = True

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

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, z, is_training=True, reuse=False):

        opt={"sn": self.sn,
             "is_training": is_training}

        with tf.variable_scope("generator", reuse=reuse):
            split_dim = self.z_dim // self.depth
            z_split = tf.split(z, num_or_size_splits=[split_dim] * self.depth, axis=-1)

            if   self.img_size==64: block_info={"counts": [1, 1, 1, 1], "sa_index": 3}
            elif self.img_size==128: block_info={"counts": [1, 1, 1, 1, 1], "sa_index": 4}
            elif self.img_size==256: block_info={"counts": [1, 2, 1, 1, 1], "sa_index": 4}
            elif self.img_size==512: block_info={"counts": [1, 2, 1, 1, 2], "sa_index": 3}
            else: raise ValueError("Invalid image size specified: "+str(self.img_size))

            ch_mul = 2**(len(block_info["counts"])-1)
            ch=ch_mul*self.ch
            x=fully_connected(z_split[0], units=4*4*ch, scope='dense', opt=opt)
            x=tf.reshape(x, shape=[-1, 4, 4, ch])

            z_i = 1
            b_i = 0

            for block_count in block_info["counts"]:
                scope='resblock_up_'+str(ch_mul)
                for sb_i in range(block_count):
                    if block_count>1: scope=scope+'_'+str(sb_i)
                    x=resblock_up_condition(x, z_split[z_i], channels=ch, use_bias=False, opt=opt, scope=scope)
                    z_i+=1

                b_i+=1
                if b_i==block_info["sa_index"]:
                    x=self_attention_2(x, channels=ch, opt=opt, scope='self_attention')

                ch=ch//2
                ch_mul=ch_mul//2

            x = batch_norm(x, opt=opt)
            x = relu(x)
            x = conv(x, channels=self.c_dim, kernel=3, stride=1, pad=1, use_bias=False, opt=opt, scope='G_logit')

            x = tanh(x)

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x, is_training=True, reuse=False):

        opt = {"sn": self.sn,
               "is_training": is_training}

        with tf.variable_scope("discriminator", reuse=reuse):
            ch = self.ch

            if   self.img_size==64: block_info={"counts": [1, 1, 1, 1], "sa_index": 1}
            elif self.img_size==128: block_info={"counts": [1, 1, 1, 1, 1], "sa_index": 1}
            elif self.img_size==256: block_info={"counts": [1, 1, 1, 2, 1], "sa_index": 2}
            elif self.img_size==512: block_info={"counts": [1, 2, 1, 1, 2], "sa_index": 2}
            else: raise ValueError("Invalid image size specified: "+str(self.img_size))


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

                ch=ch*2
                ch_mul=ch_mul*2

            ch=ch//2  # last layer has same width as previous one

            x = resblock(x, channels=ch, use_bias=False, opt=opt, scope='resblock')
            x = relu(x)

            x = global_sum_pooling(x)

            x = fully_connected(x, units=1, opt=opt, scope='D_logit')

            return x

    def gradient_penalty(self, real, fake):
        if self.gan_type.__contains__('dragan'):
            eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

            fake = real + 0.5 * x_std * eps

        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = real + alpha * (fake - real)

        logit = self.discriminator(interpolated, reuse=True)

        grad = tf.gradients(logit, interpolated)[0]  # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=1)  # l2 norm

        GP = 0

        # WGAN - LP
        if self.gan_type == 'wgan-lp':
            GP = self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
            GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        # images
        Image_Data_Class = ImageData(self.img_size, self.c_dim, self.custom_dataset)
        inputs = tf.data.Dataset.from_tensor_slices(self.data)

        gpu_device = '/gpu:0'
        inputs = inputs.\
            apply(shuffle_and_repeat(self.dataset_num)).\
            apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).\
            apply(prefetch_to_device(gpu_device, self.batch_size))

        inputs_iterator = inputs.make_one_shot_iterator()

        self.inputs = inputs_iterator.get_next()

        # noises
        self.z = tf.truncated_normal(shape=[self.batch_size, 1, 1, self.z_dim], name='random_z')

        """ Loss Function """
        # output of D for real images
        real_logits = self.discriminator(self.inputs)

        # output of D for fake images
        fake_images = self.generator(self.z)
        fake_logits = self.discriminator(fake_images, reuse=True)

        if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan':
            GP = self.gradient_penalty(real=self.inputs, fake=fake_images)
        else:
            GP = 0

        # get loss for discriminator
        self.d_loss = discriminator_loss(self.gan_type, real=real_logits, fake=fake_logits) + GP

        # get loss for generator
        self.g_loss = generator_loss(self.gan_type, fake=fake_logits, real=real_logits)

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.d_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_vars)

            self.opt = MovingAverageOptimizer(tf.train.AdamOptimizer(self.g_learning_rate, beta1=self.beta1, beta2=self.beta2), average_decay=self.moving_decay)

            self.g_optim = self.opt.minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        self.d_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_sum = tf.summary.scalar("g_loss", self.g_loss)

    ##################################################################################
    # Train
    ##################################################################################

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = self.opt.swapping_saver()

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
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, self.iteration):

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss])
                self.writer.add_summary(summary_str, counter)

                # update G network
                g_loss = None
                if (counter - 1) % self.n_critic == 0:
                    _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss])
                    self.writer.add_summary(summary_str, counter)
                    past_g_loss = g_loss

                # display training status
                counter += 1
                if g_loss == None:
                    g_loss = past_g_loss
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                sys.stdout.flush()

                # save training results every X steps
                if np.mod(idx + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

                # generate samples every X steps
                if np.mod(idx + 1, self.print_freq) == 0:
                    samples = self.sess.run(self.fake_images)
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :],
                                [manifold_h, manifold_w],
                                './' + self.sample_dir + '/' + self.model_name + '_train_{:02d}_{:05d}.png'.format(
                                    epoch, idx + 1))

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

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
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
