import scipy.misc
import numpy as np
import os
import imageio
import csv
from glob import glob
from ops import gan_dtype

import tensorflow as tf
import tensorflow.contrib.slim as slim

class ImageData:

    def __init__(self, load_size, channels, custom_dataset, flip):
        self.load_size = load_size
        self.channels = channels
        self.custom_dataset = custom_dataset
        self.flip = flip

    def image_processing(self, filename):

        if not self.custom_dataset :
            x_decode = filename
        else :
            x = tf.read_file(filename)
            x_decode = tf.image.decode_png(x, channels=self.channels)

        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        if self.flip:
            img = tf.image.random_flip_left_right(img)
        img = tf.cast(img, gan_dtype) / 127.5 - 1

        return img

    def image_processing_with_labels(self, filename, label):

        img = self.image_processing(filename)
        return img, tf.cast(label, gan_dtype)


def read_labels(path):
    labels={}
    with open(path, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter='\t', quotechar='"')

        for row in reader:
            filename = row.pop(0)
            tags = list(map(float, row))
            labels[filename] = tags

    return labels

def read_vectors(path):
    vectors=[]
    cmds={}
    with open(path, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter='\t', quotechar='"')

        for i, row in enumerate(reader):
            if len(row)>0 and row[0][0]=='!':
                row[0] = row[0][1:]
                for cmd in row:
                    p = cmd.split('=', 1)
                    cmds[p[0]]=p[1] if len(p)>1 else ''
            else:
                vec = list(map(float, row))
                vectors.append(vec)

    return cmds, vectors

def read_weights(path):
    weights=read_labels(path)
    for fn in weights:
        weights[fn]=weights[fn][0]

    return weights

def load_data(dataset_name, label_file, weight_file=None) :
    if dataset_name == 'mnist' :
        x = load_mnist()
    elif dataset_name == 'cifar10' :
        x = load_cifar10()
    else :
        x = glob(os.path.join("./dataset", dataset_name, '*.*'))

    if weight_file:
        new_x = []
        weights = read_weights(weight_file)
        for full in x:
            fn = os.path.basename(full)

            if fn not in weights:
                w = 1.0
            else:
                w = weights[fn]

            iw = int(w)
            if float(w)!=w:
                ifrac = int((w-iw)*1000.0)
                if np.random.randint(1000)<ifrac:
                    iw += 1
            for i in range(0, iw):
                new_x.append(full)

        x = new_x

    if label_file:
        labels = read_labels(label_file)
        used_labels = []
        for full in x:
            fn = os.path.basename(full)
            if fn not in labels:
                raise RuntimeError("No label found for file "+fn)
            used_labels.append(labels[fn])
    else: used_labels = None

    return x, used_labels


def preprocessing(x, size):
    x = scipy.misc.imread(x, mode='RGB')
    x = scipy.misc.imresize(x, [size, size])
    x = normalize(x)
    return x

def normalize(x) :
    return x/127.5 - 1

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    return imageio.imwrite(path, merge(images, size))


def inverse_transform(images):
    return (images+1.)/2.


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

##################################################################################
# Regularization
##################################################################################

def cosine_similarity(a, b):
    norm_a = tf.nn.l2_normalize(a, 1)
    norm_b = tf.nn.l2_normalize(b, 1)
    return tf.matmul(norm_a, norm_b, transpose_b=True)

def orthogonal_regularizer(scale, type='ortho') :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""

    def ortho_reg(w) :
        """ Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
        _, _, _, c = w.get_shape().as_list()

        w = tf.reshape(w, [-1, c])

        """ Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)

        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        if type=='ortho':
            reg = tf.subtract(w_mul, identity)
        elif type=='ortho_cosine':
            reg = cosine_similarity(w_mul, tf.ones(identity.get_shape()) - identity)
        else:
            raise ValueError("Unknown regularization method.")

        """Calculating the Loss Obtained"""
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg

def orthogonal_regularizer_fc(scale, type='ortho') :

    def ortho_reg_fully(w) :
        """ Reshaping the matrix in to 2D tensor for enforcing orthogonality"""
        _, c = w.get_shape().as_list()

        """Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        if type=='ortho':
            reg = tf.subtract(w_mul, identity)
        elif type=='ortho_cosine':
            reg = cosine_similarity(w_mul, tf.ones(identity.get_shape()) - identity)
        else:
            raise ValueError("Unknown regularization method.")

        """ Calculating the Loss """
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg_fully

def parse_int_list(str):
    if str=='none' or str=='': return []
    return [int(x.strip()) for x in str.split(",")]


def virtual_batch_steps(opt, loss, vars, virtual_batches):
    grad_factor = tf.constant(1.0/virtual_batches, dtype=gan_dtype)
    acc_vars = [tf.Variable(tf.zeros_like(v.read_value()), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]) for v in vars]
    zero_ops = [v.assign(tf.zeros_like(v)) for v in acc_vars]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        grads = opt.compute_gradients(loss, var_list=vars)
    acc_ops = [acc_vars[i].assign_add(gv[0]) for i, gv in enumerate(grads)]
    step = opt.apply_gradients([(acc_vars[i]*grad_factor, gv[1]) for i, gv in enumerate(grads)])
    return zero_ops, acc_ops, step

def create_train_ops(opt, loss, vars, virtual_batches):
    ops = {}
    ops["losses"] = {}
    ops["steps"] = virtual_batches

    if virtual_batches==1:
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            ops["init_op"] = None
            ops["step_ops"] = [opt.minimize(loss, var_list=vars)]
            ops["final_op"] = None
    else:
        zero_ops, acc_ops, step = virtual_batch_steps(opt, loss, vars, virtual_batches)
        ops["init_op"] = zero_ops
        ops["step_ops"] = acc_ops
        ops["final_op"] = step
    return ops


def run_ops(sess, ops, tensors=[], feed_dict=None, create_summaries=False):

    if tensors==None:
        tensors = []

    if ops["init_op"]!=None:
        sess.run(ops["init_op"], feed_dict=feed_dict)

    if create_summaries:
        if "summaries" not in ops:
            ops["summaries"] = {}

    losses = []
    loss_names = []
    loss_sums = []
    summaries = []
    for key, loss in ops["losses"].items():
        losses.append(loss)
        loss_names.append(key)
        loss_sums.append(0)

        if create_summaries:
            if key not in ops["summaries"]:
                ops["summaries"][key] = tf.summary.scalar(key, loss)
            summaries.append(ops["summaries"][key])

    for i in range(ops["steps"]):
        results = list(sess.run(losses+summaries+tensors+ops["step_ops"], feed_dict=feed_dict))

        summary_idx = len(losses)
        tensor_idx = summary_idx+len(summaries)
        op_idx = tensor_idx+len(tensors)

        loss_values = results[:summary_idx]
        summary_results = results[summary_idx:tensor_idx]
        tensor_results = results[tensor_idx:op_idx]

        for i, loss in enumerate(loss_values):
            loss_sums[i] += loss


    if ops["final_op"]!=None:
        sess.run(ops["final_op"])

    loss_values = {}
    for i in range(len(loss_sums)):
        loss_values[loss_names[i]] = loss_sums[i]/ops["steps"]

    # todo: merge tensor results if steps>1

    return loss_values, tensor_results, summary_results

def create_hist_summaries():
    summaries = []
    for var in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES):
        if "/Adam" in var.name or "ExponentialMovingAverage" in var.name:
            continue
        if var.name.startswith("beta1_power") or var.name.startswith("beta2_power"):
            continue

        hist = tf.summary.histogram(var.name+'/hist', tf.reshape(var,[-1]))
        summaries += [hist]

    return tf.summary.merge(summaries)

def round_up(val, multiple):
    return (int(val) + multiple - 1) // multiple * multiple