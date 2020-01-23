import tensorflow as tf

##################################################################################
# Initialization
##################################################################################

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# Truncated_normal : tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
# Orthogonal : tf.orthogonal_initializer(1.0) / relu = sqrt(2), the others = 1.0

weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
gan_dtype = tf.float32

##################################################################################
# Layer
##################################################################################

# pad = ceil[ (kernel - stride) / 2 ]

def conv(x, channels, opt, kernel=4, stride=2, pad=0, dilation=1, pad_type='reflect', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope) as full_scope:
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero' :
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect' :
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if opt["sn"]:
            if 'generator' in full_scope.name:
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                    regularizer=opt["conv_regularizer"], dtype=gan_dtype)
            else :
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                    regularizer=None, dtype=gan_dtype)

            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID', dilations=dilation)
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0), dtype=gan_dtype)
                x = tf.nn.bias_add(x, bias)

        else :
            if 'generator' in full_scope.name:
                x = tf.layers.conv2d(inputs=x, filters=channels,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=opt["conv_regularizer"],
                                     strides=stride, use_bias=use_bias, dilations=dilation)
            else :
                x = tf.layers.conv2d(inputs=x, filters=channels,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=None,
                                     strides=stride, use_bias=use_bias, dilations=dilation)


        return x


def deconv(x, channels, opt, kernel=4, stride=2, padding='SAME', use_bias=True, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        else:
            output_shape = [x_shape[0], x_shape[1] * stride + max(kernel - stride, 0), x_shape[2] * stride + max(kernel - stride, 0), channels]

        if opt["sn"]:
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=opt["conv_regularizer"], dtype=gan_dtype)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0), dtype=gan_dtype)
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=opt["conv_regularizer"],
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x

def fully_connected(x, units, opt, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope) as full_scope:
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if opt["sn"]:
            if 'generator' in full_scope.name:
                w = tf.get_variable("kernel", [channels, units], gan_dtype, initializer=weight_init, regularizer=opt["fc_regularizer"])
            else :
                w = tf.get_variable("kernel", [channels, units], gan_dtype, initializer=weight_init, regularizer=None)

            if use_bias :
                bias = tf.get_variable("bias", [units], initializer=tf.constant_initializer(0.0), dtype=gan_dtype)

                x = tf.matmul(x, spectral_norm(w)) + bias
            else :
                x = tf.matmul(x, spectral_norm(w))

        else :
            if 'generator' in full_scope.name:
                x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                    kernel_regularizer=opt["fc_regularizer"], use_bias=use_bias)
            else :
                x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                    kernel_regularizer=None, use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

##################################################################################
# Residual-block, Self-Attention-block
##################################################################################

def resblock(x_init, channels, opt, use_bias=True, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, opt=opt)
            if (opt["bn_in_d"]): x = batch_norm(x, opt=opt)
            x = opt["act"](x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, opt=opt)
            if (opt["bn_in_d"]): x = batch_norm(x, opt=opt)

        return x + x_init

def upconv(x, channels, opt, use_bias=True):
    if opt["upsampling_method"] == 'deconv3':
        return deconv(x, channels, kernel=3, stride=2, use_bias=use_bias, opt=opt)
    elif opt["upsampling_method"] == 'deconv4':
        return deconv(x, channels, kernel=4, stride=2, use_bias=use_bias, opt=opt)
    elif opt["upsampling_method"] == 'resize_conv':
        x = up_sample(x, 2)
        x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, opt=opt)
        return x

    else: raise ValueError("Invalid upsampling method specified: "+str(opt["upsampling_method"]))

def g_conv(x, channels, opt, use_bias=True):
    if opt["g_conv"] == 'deconv3':
        return deconv(x, channels, kernel=3, stride=1, use_bias=use_bias, opt=opt)
    elif opt["g_conv"] == 'deconv4':
        return deconv(x, channels, kernel=4, stride=1, use_bias=use_bias, opt=opt)
    elif opt["g_conv"] == 'conv3':
        return conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, opt=opt)
    elif opt["g_conv"] == 'conv5':
        return conv(x, channels, kernel=5, stride=1, pad=2, use_bias=use_bias, opt=opt)

    else: raise ValueError("Invalid generator convolution type specified: "+str(opt["g_conv"]))

def resblock_up(x_init, channels, opt, use_bias=True, scope='resblock_up'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = batch_norm(x_init, opt=opt)
            x = opt["act"](x)
            x = upconv(x, channels, use_bias=use_bias, opt=opt)

        with tf.variable_scope('res2') :
            x = batch_norm(x, opt=opt)
            x = opt["act"](x)
            x = g_conv(x, channels, use_bias=use_bias, opt=opt)

        with tf.variable_scope('skip') :
            x_init = upconv(x_init, channels, use_bias=use_bias, opt=opt)


    return x + x_init

def resblock_up_condition(x_init, z, channels, opt, use_bias=True, scope='resblock_up'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = condition_batch_norm(x_init, z, opt=opt)
            x = opt["act"](x)
            x = upconv(x, channels, use_bias=use_bias, opt=opt)

        with tf.variable_scope('res2') :
            x = condition_batch_norm(x, z, opt=opt)
            x = opt["act"](x)
            x = g_conv(x, channels, use_bias=use_bias, opt=opt)

        with tf.variable_scope('skip') :
            x_init = upconv(x_init, channels, use_bias=use_bias, opt=opt)


    return x + x_init


def resblock_down(x_init, channels, opt, use_bias=True, scope='resblock_down'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            if (opt["bn_in_d"]): x = batch_norm(x_init, opt=opt)
            else: x = x_init
            x = opt["act"](x)
            x = conv(x, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, opt=opt)

        with tf.variable_scope('res2') :
            if (opt["bn_in_d"]): x = batch_norm(x, opt=opt)
            x = opt["act"](x)
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, opt=opt)

        with tf.variable_scope('skip') :
            x_init = conv(x_init, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, opt=opt)


    return x + x_init

def clown_conv(x, channels, opt, use_bias=True, scope='clown', z=None):

    split_ch = channels//8
    rest_split = channels - split_ch*7
    deconv4_ch = split_ch + rest_split

    with tf.variable_scope(scope):

        deconv4 = deconv(x, deconv4_ch, kernel=4, stride=1, use_bias=use_bias, scope="deconv4", opt=opt)
        deconv3 = deconv(x, split_ch * 2, kernel=3, stride=1, use_bias=use_bias, scope="deconv3", opt=opt)
        deconv2 = deconv(x, split_ch, kernel=2, stride=1, use_bias=use_bias, scope="deconv2", opt=opt)
        conv3 = conv(x, split_ch, kernel=3, stride=1, pad=1, use_bias=use_bias, scope="conv3", opt=opt)
        conv5 = conv(x, split_ch, kernel=5, stride=1, pad=2, use_bias=use_bias, scope="conv5", opt=opt)
        dilconv5 = conv(x, split_ch, kernel=5, stride=1, pad=4, dilation=2, use_bias=use_bias, scope="dilconv5", opt=opt)

        concat = tf.concat([deconv4, deconv3, deconv2, conv3, conv5, dilconv5], axis=-1)
        if z!=None:
            concat = condition_batch_norm(concat, z, opt=opt)
        else:
            concat = batch_norm(concat, opt=opt)
        concat = prelu(concat)

        return concat


def self_attention(x, channels, opt, scope='self_attention'):
    with tf.variable_scope(scope):
        f = conv(x, channels // 8, kernel=1, stride=1, opt=opt, scope='f_conv')  # [bs, h, w, c']
        g = conv(x, channels // 8, kernel=1, stride=1, opt=opt, scope='g_conv')  # [bs, h, w, c']
        h = conv(x, channels, kernel=1, stride=1, opt=opt, scope='h_conv')  # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0), dtype=gan_dtype)

        o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
        x = gamma * o + x

    return x

def self_attention_2(x, channels, opt, scope='self_attention'):
    with tf.variable_scope(scope):
        f = conv(x, channels // 8, kernel=1, stride=1, opt=opt, scope='f_conv')  # [bs, h, w, c']
        f = max_pooling(f)

        g = conv(x, channels // 8, kernel=1, stride=1, opt=opt, scope='g_conv')  # [bs, h, w, c']

        h = conv(x, channels // 2, kernel=1, stride=1, opt=opt, scope='h_conv')  # [bs, h, w, c]
        h = max_pooling(h)

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0), dtype=gan_dtype)

        o = tf.reshape(o, shape=[x.shape[0], x.shape[1], x.shape[2], channels // 2])  # [bs, h, w, C]
        o = conv(o, channels, kernel=1, stride=1, opt=opt, scope='attn_conv')
        x = gamma * o + x

    return x

##################################################################################
# Sampling
##################################################################################

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])

    return gap

def global_sum_pooling(x) :
    gsp = tf.reduce_sum(x, axis=[1, 2])

    return gsp

def max_pooling(x) :
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')
    return x

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)

def prelu(x, scope=None, init_val=0.0):
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        alphas = tf.get_variable('alpha', x.get_shape()[-1], initializer=tf.constant_initializer(init_val), dtype=gan_dtype)
        pos = tf.nn.relu(x)
        neg = alphas * (x - abs(x)) * 0.5
        return pos + neg

def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, opt, scope='batch_norm'):
    return tf.layers.batch_normalization(x,
                                         momentum=0.98,
                                         epsilon=1e-05,
                                         training=opt["is_training"],
                                         name=scope)

def condition_batch_norm(x, z, opt, scope='batch_norm'):
    with tf.variable_scope(scope) :
        _, _, _, c = x.get_shape().as_list()
        decay = 0.98
        epsilon = 1e-05

        test_mean = tf.get_variable("pop_mean", shape=[c], dtype=gan_dtype, initializer=tf.constant_initializer(0.0), trainable=False)
        test_var = tf.get_variable("pop_var", shape=[c], dtype=gan_dtype, initializer=tf.constant_initializer(1.0), trainable=False)

        beta = fully_connected(z, units=c, scope='beta', opt=opt)
        gamma = fully_connected(z, units=c, scope='gamma', opt=opt)

        beta = tf.reshape(beta, shape=[-1, 1, 1, c])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, c])

        if opt["is_training"]:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            ema_mean = tf.assign(test_mean, test_mean * decay + batch_mean * (1 - decay))
            ema_var = tf.assign(test_var, test_var * decay + batch_var * (1 - decay))

            with tf.control_dependencies([ema_mean, ema_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)
        else:
            return tf.nn.batch_normalization(x, test_mean, test_var, beta, gamma, epsilon)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False, dtype=gan_dtype)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(loss_func, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if loss_func == 'ra-gan' or loss_func == 'ra-dragan':
        d_xr = real - tf.reduce_mean(fake)
        d_xf = fake - tf.reduce_mean(real)
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=d_xr))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=d_xf))

    if loss_func == 'ra-hinge':
        d_xr = real - tf.reduce_mean(fake)
        d_xf = fake - tf.reduce_mean(real)
        real_loss = tf.reduce_mean(relu(1.0 - d_xr))
        fake_loss = tf.reduce_mean(relu(1.0 + d_xf))

    if loss_func == 'hinge' :
        real_loss = tf.reduce_mean(relu(1.0 - real))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss

def generator_loss(loss_func, fake, real):
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if loss_func == 'gan' or loss_func == 'dragan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'ra-gan' or loss_func == 'ra-dragan':
        d_xr = real - tf.reduce_mean(fake)
        d_xf = fake - tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=d_xf))
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real), logits=d_xr))
        fake_loss = fake_loss + real_loss

    if loss_func == 'ra-hinge':
        d_xr = real - tf.reduce_mean(fake)
        d_xf = fake - tf.reduce_mean(real)
        real_loss = tf.reduce_mean(relu(1.0 - d_xf))
        fake_loss = tf.reduce_mean(relu(1.0 + d_xr)) + real_loss

    if loss_func == 'hinge' :
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss
