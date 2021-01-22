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

from utils import round_up

##################################################################################
# Layer
##################################################################################


def subpixel_conv(x, channels, opt, kernel=3, scale=2, use_bias=True, scope='subpixel_conv_0'):
    r2 = scale*scale
    x = conv(x, channels * r2, opt=opt, kernel=kernel, stride=1, use_bias=use_bias, pad=(kernel-1)/2.0, scope=scope)
    x = tf.nn.depth_to_space(x, block_size=scale)
    return x

def decode_kernel_sizes(str):
    parts = str.split(',')
    slices = []
    sum = 0
    for part in parts:
        size, kernel_size = part.split('x')
        slices+=[{"size": int(size), "kernel": int(kernel_size)}]
        sum+=int(size)

    ret = {}
    ret["slices"] = slices
    ret["total_channels"] = sum

    return ret

def encode_kernel_sizes(slices, ch_mul=1.0):
    return ",".join([str(int(float(slice["size"])*ch_mul+0.00000001))+"x"+str(slice["kernel"]) for slice in slices])



def conv(x, channels, opt, kernel=4, stride=2, pad=0, dilation=1, use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope) as full_scope:

        if isinstance(kernel, str):
            slices = decode_kernel_sizes(kernel)["slices"]
            slice_convs = []
            for slice in slices:
                slice_conv = conv(x, slice["size"], opt, kernel=slice["kernel"], stride=stride, pad=(slice["kernel"]-1)//2, dilation=dilation, use_bias=use_bias, scope='conv'+str(slice["kernel"])+"_slice")
                slice_convs += [slice_conv]

            return tf.concat(slice_convs, axis=-1)



        tf_pad_type = 'VALID'

        if pad > 0:
            pad_type = opt.get("conv", {}).get("padding_type", 'reflect')
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = int(pad//2)
            pad_bottom = int(pad - pad_top)
            pad_left = int(pad//2)
            pad_right = int(pad - pad_left)


            if pad_type == 'zero':
                tf_pad_type = 'SAME'
            elif pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
            else:
                raise ValueError("Unsupported padding type: "+str(pad_type))

        if opt.get("conv", {}).get("sn", True):
            if 'generator' in full_scope.name:
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                    regularizer=opt.get("conv", {}).get("regularizer", None), dtype=gan_dtype)
            else :
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                    regularizer=None, dtype=gan_dtype)

            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding=tf_pad_type, dilations=dilation)
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0), dtype=gan_dtype)
                x = tf.nn.bias_add(x, bias)

        else :
            if 'generator' in full_scope.name:
                x = tf.layers.conv2d(inputs=x, filters=channels,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=opt.get("conv", {}).get("regularizer", None),
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

        if opt.get("conv", {}).get("sn", True):
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=opt.get("conv", {}).get("regularizer", None), dtype=gan_dtype)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0), dtype=gan_dtype)
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=opt.get("conv", {}).get("regularizer", None),
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x

def get_variable_with_custom_lr(name, shape, regularizer, lrmul):
    w = tf.get_variable(name, shape, gan_dtype, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02/lrmul), regularizer=regularizer)
    if (lrmul==1.0):
        return w
    else:
        return w * lrmul

def fully_connected(x, units, opt, use_bias=True, lrmul=1.0, scope='fully_0'):
    with tf.variable_scope(scope) as full_scope:
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if opt.get("conv", {}).get("sn", True):
            if 'generator' in full_scope.name:
                w = get_variable_with_custom_lr("kernel", shape=[channels, units], regularizer=opt["fc_regularizer"], lrmul=lrmul)
            else :
                w = get_variable_with_custom_lr("kernel", shape=[channels, units], regularizer=None, lrmul=lrmul)

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
            if (opt["bn_in_d"]): x = bn(x, opt=opt)
            x = opt["act"](x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, opt=opt)
            if (opt["bn_in_d"]): x = bn(x, opt=opt)

        return x + x_init

def upconv(x, channels, opt, use_bias=True):
    if opt["upsampling_method"] == 'deconv3':
        return deconv(x, channels, kernel=3, stride=2, use_bias=use_bias, opt=opt)
    elif opt["upsampling_method"] == 'deconv4':
        return deconv(x, channels, kernel=4, stride=2, use_bias=use_bias, opt=opt)
    elif opt["upsampling_method"] == 'deconv6':
        return deconv(x, channels, kernel=6, stride=2, use_bias=use_bias, opt=opt)
    elif opt["upsampling_method"] == 'subpixel2':
        return subpixel_conv(x, channels, kernel=2, scale=2, use_bias=use_bias, opt=opt)
    elif opt["upsampling_method"] == 'subpixel3':
        return subpixel_conv(x, channels, kernel=3, scale=2, use_bias=use_bias, opt=opt)
    elif opt["upsampling_method"] == 'resize_conv':
        x = up_sample(x, 2)
        x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, opt=opt)
        return x
    elif opt["upsampling_method"]=='nn':
        return up_sample(x, 2)

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
            x = bn(x_init, opt=opt)
            x = opt["act"](x)
            x = upconv(x, channels, use_bias=use_bias, opt=opt)

        with tf.variable_scope('res2') :
            x = bn(x, opt=opt)
            x = opt["act"](x)
            x = g_conv(x, channels, use_bias=use_bias, opt=opt)

        with tf.variable_scope('skip') :
            x_init = upconv(x_init, channels, use_bias=use_bias, opt=opt)


    return x + x_init

def resblock_up_condition(x_init, z, channels, opt, use_bias=True, scope='resblock_up'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = cond_bn(x_init, z, opt=opt)
            x = opt["act"](x)
            x = upconv(x, channels, use_bias=use_bias, opt=opt)

        with tf.variable_scope('res2') :
            x = cond_bn(x, z, opt=opt)
            x = opt["act"](x)
            x = g_conv(x, channels, use_bias=use_bias, opt=opt)

        with tf.variable_scope('skip') :
            x_init = upconv(x_init, channels, use_bias=use_bias, opt=opt)


    return x + x_init


def downconv(x, channels, opt, use_bias=True, method=None):
    if method==None:
        method = opt["downsampling_method"]

    if method == 'strided_conv3':
        return conv(x, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, opt=opt)
    elif method == 'resize_conv1':
        x = conv(x, channels, kernel=1, stride=1, pad=0, use_bias=use_bias, opt=opt)
        return avg_pooling(x)
    elif method == 'resize_conv3':
        x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, opt=opt)
        return avg_pooling(x)
    elif method == 'pool_only':
        return avg_pooling(x)
    elif method == 'max_pool_only':
        return max_pooling(x)

    else: raise ValueError("Invalid downsampling method specified: "+str(method))

def resblock_down(x_init, channels, opt, use_bias=True, scope='resblock_down'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            if (opt["bn_in_d"]): x = bn(x_init, opt=opt)
            else: x = x_init
            x = opt["act"](x)
            res_method = opt["downsampling_method"]
            if res_method!='strided_conv3':
                res_method = 'resize_conv3'
            x = downconv(x, channels, use_bias=use_bias, opt=opt, method=res_method)

        with tf.variable_scope('res2') :
            if (opt["bn_in_d"]): x = bn(x, opt=opt)
            x = opt["act"](x)
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, opt=opt)

        with tf.variable_scope('skip') :
            x_init = downconv(x_init, channels, use_bias=use_bias, opt=opt, method=opt["downsampling_method"])


    return x + x_init



def resblock_up_cond_deep(x_init, z, channels_out, opt, upscale=True, use_bias=True, scope='deep_resblock'):

    channels_in = int(x_init.get_shape()[-1])
    inner_channels = round_up((channels_in + channels_out)//6, 8)

    with tf.variable_scope(scope):
        with tf.variable_scope('bottleneck'):
            x = cond_bn(x_init, z, opt=opt)
            x = opt["act"](x)
            x = conv(x, inner_channels, kernel=1, stride=1, use_bias=False, opt=opt)

        with tf.variable_scope('upscale'):
            x = cond_bn(x, z, opt=opt)
            x = opt["act"](x)
            if upscale:
                x = upconv(x, inner_channels, use_bias=False, opt=opt)

        with tf.variable_scope('inner1'):
            x = g_conv(x, inner_channels, use_bias=False, opt=opt)
            x = cond_bn(x, z, opt=opt)
            x = opt["act"](x)

        with tf.variable_scope('inner2'):
            x = g_conv(x, inner_channels, use_bias=False, opt=opt)
            x = bn(x, opt=opt)
            x = opt["act"](x)

        with tf.variable_scope('proj'):
            x = conv(x, channels_out, kernel=1, stride=1, use_bias=use_bias, opt=opt)

        with tf.variable_scope('skip'):
            if channels_in != channels_out:
                print(inner_channels, channels_in, channels_out, channels_in - channels_out)
                kept, dropped = tf.split(x_init, num_or_size_splits=[channels_out, channels_in - channels_out], axis=-1)
            else:
                kept = x_init

            if upscale:
                x_init = upconv(kept, channels_out, use_bias=use_bias, opt=opt)


    return x + x_init


def resblock_down_deep(x_init, channels_out, opt, downscale=True, use_bias=True, scope='deep_resblock'):

    channels_in = x_init.get_shape()[-1]
    inner_channels = round_up((channels_in + channels_out)//6, 8)

    with tf.variable_scope(scope):
        with tf.variable_scope('bottleneck'):
            x = x_init
            if (opt["bn_in_d"]): x = bn(x, opt=opt)
            x = opt["act"](x)
            x = conv(x, inner_channels, kernel=1, stride=1, pad=0, use_bias=use_bias, opt=opt)

        with tf.variable_scope('inner1'):
            if (opt["bn_in_d"]): x = bn(x, opt=opt)
            x = opt["act"](x)
            x = conv(x, inner_channels, kernel=3, stride=1, pad=1, use_bias=use_bias, opt=opt)

        with tf.variable_scope('inner2'):
            if (opt["bn_in_d"]): x = bn(x, opt=opt)
            x = opt["act"](x)
            x = conv(x, inner_channels, kernel=3, stride=1, pad=1, use_bias=use_bias, opt=opt)

        with tf.variable_scope('downscale'):
            x = opt["act"](x)

            if downscale:
                x = downconv(x, inner_channels, use_bias=use_bias, opt=opt, method='pool_only')

        with tf.variable_scope('proj'):
            x = conv(x, channels_out, kernel=1, stride=1, pad=0, use_bias=use_bias, opt=opt)

        with tf.variable_scope('skip'):
            if downscale:
                x_init = downconv(x_init, channels_in, use_bias=use_bias, opt=opt, method='pool_only')
            if channels_in != channels_out:
                conv_ch = channels_out - channels_in
                dense = conv(x_init, conv_ch, kernel=1, stride=1, pad=0, use_bias=use_bias, opt=opt)
                x_init = tf.concat([x_init, dense], axis=-1)


    return x + x_init

def clown_conv(x, channels, opt, use_bias=True, scope='clown', z=None):

    split_ch = channels//8
    half_split_ch = split_ch//2
    other_half_split_ch = split_ch - half_split_ch
    rest_split = channels - split_ch*7
    deconv4_ch = split_ch + rest_split
    conv5_ch = split_ch

    no_deconv2 = opt.get("mixed_conv_no_deconv2", False)

    if no_deconv2:
        deconv4_ch += half_split_ch
        conv5_ch += other_half_split_ch

    with tf.variable_scope(scope):

        splits = []
        splits.append(deconv(x, deconv4_ch, kernel=4, stride=1, use_bias=use_bias, scope="deconv4", opt=opt))
        splits.append(deconv(x, split_ch * 2, kernel=3, stride=1, use_bias=use_bias, scope="deconv3", opt=opt))
        if not no_deconv2:
            splits.append(deconv(x, split_ch, kernel=2, stride=1, use_bias=use_bias, scope="deconv2", opt=opt))
        splits.append(conv(x, split_ch, kernel=3, stride=1, pad=1, use_bias=use_bias, scope="conv3", opt=opt))
        splits.append(conv(x, conv5_ch, kernel=5, stride=1, pad=2, use_bias=use_bias, scope="conv5", opt=opt))
        splits.append(conv(x, split_ch, kernel=5, stride=1, pad=4, dilation=2, use_bias=use_bias, scope="dilconv5", opt=opt))

        concat = tf.concat(splits, axis=-1)
        if z!=None:
            concat = cond_bn(concat, z, opt=opt)
        else:
            concat = bn(concat, opt=opt)
        concat = prelu(concat)

        return concat

def mixed_resblock(x, inner_channels, out_channels, opt, use_bias=False, z=None, scope='res_mixed'):
    with tf.variable_scope(scope):
        res = clown_conv(x, inner_channels, scope="clown", opt=opt, z=z)
        res = conv(res, channels=out_channels, kernel=1, stride=1, pad=0, use_bias=False, opt=opt, scope='proj')
    return x + res


def self_attention(x, channels, opt, scope='self_attention'):
    with tf.variable_scope(scope):

        use_bias = opt.get("self_attention_bias", False)

        f = conv(x, channels // 8, kernel=1, stride=1, opt=opt, scope='f_conv', use_bias=use_bias)  # [bs, h, w, c']
        g = conv(x, channels // 8, kernel=1, stride=1, opt=opt, scope='g_conv', use_bias=use_bias)  # [bs, h, w, c']
        h = conv(x, channels, kernel=1, stride=1, opt=opt, scope='h_conv', use_bias=use_bias)  # [bs, h, w, c]

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

        use_bias = opt.get("self_attention_bias", False)

        f = conv(x, channels // 8, kernel=1, stride=1, opt=opt, scope='f_conv', use_bias=use_bias)  # [bs, h, w, c']
        f = max_pooling(f)

        g = conv(x, channels // 8, kernel=1, stride=1, opt=opt, scope='g_conv', use_bias=use_bias)  # [bs, h, w, c']

        h = conv(x, channels // 2, kernel=1, stride=1, opt=opt, scope='h_conv', use_bias=use_bias)  # [bs, h, w, c]
        h = max_pooling(h)

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0), dtype=gan_dtype)

        o = tf.reshape(o, shape=[x.shape[0], x.shape[1], x.shape[2], channels // 2])  # [bs, h, w, C]
        o = conv(o, channels, kernel=1, stride=1, opt=opt, scope='attn_conv', use_bias=use_bias)
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

def avg_pooling(x):
    x = tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')
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

def bn(x, opt={}, scope='batch_norm'):
    type = opt.get("bn",{}).get("type","bn")

    if type=='batch_norm_broken_renorm':
        type = 'batch_norm'
        if scope=='batch_norm':
            scope = 'batch_renorm'

    if type=='bn' or type=='batch_norm':
        return batch_norm(x, opt=opt, scope=scope)
    elif type=='batch_renorm':
        if scope=='batch_norm':
            scope = 'batch_renorm'
        return batch_renorm(x, opt=opt, scope=scope)
    else:
        raise ValueError("Unknown BN type: "+str(type))

def cond_bn(x, z, opt={}, scope='batch_norm'):
    type = opt.get("bn",{}).get("type","bn")

    if type=='batch_norm_broken_renorm':
        type = 'batch_norm'
        if scope=='batch_norm':
            scope = 'batch_renorm'

    if type=='bn' or type=='batch_norm':
        return condition_batch_norm(x, z, opt=opt, scope=scope)
    elif type=='batch_renorm':
        if scope=='batch_norm':
            scope = 'batch_renorm'
        return condition_batch_renorm(x, z, opt=opt, scope=scope)
    else:
        raise ValueError("Unknown BN type: "+str(type))

def batch_norm(x, opt={}, scope='batch_norm'):
    return tf.layers.batch_normalization(x,
                                         momentum=opt.get("bn", {}).get("momentum", 0.98),
                                         epsilon=1e-05,
                                         training=opt["is_training"],
                                         name=scope)

def normalize_renorm_clipping_params(renorm_clipping):
    if "rmax" not in renorm_clipping:
        renorm_clipping["rmax"] = 1.5

    if "dmax" not in renorm_clipping:
        renorm_clipping["dmax"] = 0.5

    if "rmax" in renorm_clipping and not "rmin" in renorm_clipping:
        renorm_clipping["rmin"] = 1.0/renorm_clipping["rmax"]

    return renorm_clipping


def batch_renorm(x, opt={}, scope='batch_renorm'):
    renorm_clipping = normalize_renorm_clipping_params(opt.get("bn", {}).get("renorm_clipping", {}))
    return tf.layers.batch_normalization(x,
                                         momentum=opt.get("bn", {}).get("momentum", 0.98),
                                         epsilon=1e-05,
                                         training=opt["is_training"],
                                         name=scope,
                                         renorm=True,
                                         renorm_momentum=opt.get("bn", {}).get("renorm_momentum", 0.9),
                                         renorm_clipping=renorm_clipping)

def condition_batch_norm(x, z, opt={}, scope='batch_norm'):
    with tf.variable_scope(scope) :
        _, _, _, c = x.get_shape().as_list()

        fake = False

        decay = opt.get("bn", {}).get("momentum", 0.98)
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
                if fake:
                    batch_mean = 0.0
                    batch_var = 1.0
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)
        else:
            if fake:
                test_mean = 0.0
                test_var = 1.0
            return tf.nn.batch_normalization(x, test_mean, test_var, beta, gamma, epsilon)

def condition_batch_renorm(x, z, opt={}, scope='batch_renorm'):
    with tf.variable_scope(scope) :
        _, _, _, c = x.get_shape().as_list()

        fake=False

        renorm_clipping = normalize_renorm_clipping_params(opt.get("bn", {}).get("renorm_clipping", {}))

        test_decay = opt.get("bn", {}).get("momentum", 0.98)
        renorm_decay = opt.get("bn", {}).get("renorm_momentum", 0.9)
        shared = opt.get("bn", {}).get("shared_renorm", False)
        renorm_fadein_decay = opt.get("bn", {}).get("renorm_fadein_decay", 0.9999)

        if not shared:
            test_decay = renorm_decay

        epsilon = 1e-05

        test_mean = tf.get_variable("pop_mean", shape=[c], dtype=gan_dtype, initializer=tf.constant_initializer(0.0), trainable=False)
        test_var = tf.get_variable("pop_var", shape=[c], dtype=gan_dtype, initializer=tf.constant_initializer(1.0), trainable=False)

        if not shared:
            renorm_mean = tf.get_variable("renorm_mean", shape=[c], dtype=gan_dtype, initializer=tf.constant_initializer(0.0), trainable=False)
            renorm_var = tf.get_variable("renorm_var", shape=[c], dtype=gan_dtype, initializer=tf.constant_initializer(1.0), trainable=False)
            renorm_weight = tf.get_variable("renorm_weight", shape=[], dtype=gan_dtype, initializer=tf.constant_initializer(0.0), trainable=False)
        else:
            renorm_mean = test_mean
            renorm_var = test_var
            renorm_weight = 1.0

        beta = fully_connected(z, units=c, scope='beta', opt=opt)
        gamma = fully_connected(z, units=c, scope='gamma', opt=opt)

        beta = tf.reshape(beta, shape=[-1, 1, 1, c])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, c])

        if opt["is_training"]:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])

            rmax = renorm_clipping["rmax"]
            rmin = renorm_clipping["rmin"]
            dmax = renorm_clipping["dmax"]

            sigma = tf.sqrt(batch_var + epsilon)
            renorm_sigma = tf.sqrt(renorm_var + epsilon)
            weighted_renorm_sigma = renorm_weight*renorm_sigma + (1 - renorm_weight)*sigma
            weighted_renorm_mean = renorm_weight*renorm_mean + (1 - renorm_weight)*batch_mean
            r = tf.stop_gradient(tf.clip_by_value(sigma/weighted_renorm_sigma, rmin, rmax))
            d = tf.stop_gradient(tf.clip_by_value((batch_mean - weighted_renorm_mean)/weighted_renorm_sigma, -dmax, dmax))

            test_mean_op = tf.assign(test_mean, test_mean * test_decay + batch_mean * (1 - test_decay))
            test_var_op = tf.assign(test_var, test_var * test_decay + batch_var * (1 - test_decay))

            ema_ops = [test_mean_op, test_var_op]

            if not shared:
                renorm_mean_op = tf.assign(renorm_mean, renorm_mean*renorm_decay + batch_mean*(1 - renorm_decay))
                renorm_var_op = tf.assign(renorm_var, renorm_var*renorm_decay + batch_var*(1 - renorm_decay))
                renorm_w_op = tf.assign(renorm_weight, renorm_weight*renorm_fadein_decay + 1.0*(1 - renorm_fadein_decay))
                ema_ops += [renorm_mean_op, renorm_var_op, renorm_w_op]

            with tf.control_dependencies(ema_ops):
                if fake:
                    return tf.nn.batch_normalization(x, 0.0, 1.0, beta, gamma, epsilon)
                else:
                    return tf.nn.batch_normalization(x, batch_mean, batch_var, beta + d*gamma, r*gamma, epsilon)
        else:
            if fake:
                test_mean = 0.0
                test_var = 1.0
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

def discriminator_loss(loss_func, real, fake, flood_level=0):
    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if loss_func == 'ra-lsgan' :
        d_xr = real - tf.reduce_mean(fake)
        d_xf = fake - tf.reduce_mean(real)

        real_loss = tf.reduce_mean(tf.squared_difference(d_xr, 1.0))
        fake_loss = tf.reduce_mean(tf.squared_difference(d_xf, -1.0))

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

    if flood_level:
        loss = flood_loss(loss, flood_level)

    return loss

def generator_loss(loss_func, fake, real, flood_level=0):

    fake_loss = 0
    real_loss = 0

    if loss_func.__contains__('wgan') :
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if loss_func == 'ra-lsgan' :
        d_xr = real - tf.reduce_mean(fake)
        d_xf = fake - tf.reduce_mean(real)

        real_loss = tf.reduce_mean(tf.squared_difference(d_xr, -1.0))
        fake_loss = tf.reduce_mean(tf.squared_difference(d_xf, 1.0))

    if loss_func == 'gan' or loss_func == 'dragan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'ra-gan' or loss_func == 'ra-dragan':
        d_xr = real - tf.reduce_mean(fake)
        d_xf = fake - tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=d_xf))
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real), logits=d_xr))

    if loss_func == 'ra-hinge':
        d_xr = real - tf.reduce_mean(fake)
        d_xf = fake - tf.reduce_mean(real)
        real_loss = tf.reduce_mean(relu(1.0 - d_xf))
        fake_loss = tf.reduce_mean(relu(1.0 + d_xr))

    if loss_func == 'hinge' :
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss + real_loss

    if flood_level:
        loss = flood_loss(loss, flood_level)

    return loss

def glu(x, opt=None):
    main, gate = tf.split(x, num_or_size_splits=2, axis=-1)
    gate = tf.math.sigmoid(gate)
    return main*gate

def flood_loss(loss, flood_level):
    return tf.math.abs(loss - flood_level) + flood_level
