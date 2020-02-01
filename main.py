from BigGAN import BigGAN
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of BigGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train, test, service]')
    parser.add_argument('--dataset', type=str, default='celebA-HQ', help='[mnist / cifar10 / custom_dataset]')

    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch per gpu')
    parser.add_argument('--virtual_batches', type=int, default=1, help='number of gradient accumulations per step')
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--d_ch', type=int, default=0, help='base channel number per layer in discriminator')

    parser.add_argument('--print_freq', type=int, default=250, help='how often a set of samples is generated')
    parser.add_argument('--save_freq', type=int, default=1000, help='how often checkpoints are saved')
    parser.add_argument('--histogram_freq', type=int, default=125, help='how often histogram summaries are saved, 0 to deactivate')
    parser.add_argument('--keep_checkpoints', type=int, default=5, help='keep N last checkpoints, 0 to keep all')

    parser.add_argument('--g_lr', type=float, default=0.00005, help='learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for discriminator')

    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')
    parser.add_argument('--moving_decay', type=float, default=0.999 , help='moving average decay for generator')

    parser.add_argument('--z_dim', type=int, default=256, help='dimension of noise vector')
    parser.add_argument('--shared_z', type=int, default=0, help='number of z values to share across all layers - the remaining values are distributed between all layers')
    parser.add_argument('--c_dim', type=int, default=3, help='number of color channels [1, 3, 4]')
    parser.add_argument('--alpha_mask', type=str2bool, default=True, help='if c_dim is 4, multiplies the color channels with the alpha before feeding the image to the discriminator')
    parser.add_argument('--g_alpha_helper', type=str2bool, default=True, help='adds color channels to alpha channel to help with early training; best used together with alpha masking')
    parser.add_argument('--first_split_ratio', type=int, default=3, help='ratio of z values assigned to the first layer')
    parser.add_argument('--z_reconstruct', type=str2bool, default=False, help='train the discriminator to reconstruct z')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')
    parser.add_argument('--bn_in_d', type=str2bool, default=False, help='whether to use BN in the discriminator')
    parser.add_argument('--bn_type', type=str, default='batch_norm', help='[batch_norm / batch_renorm]')
    parser.add_argument('--bn_momentum', type=float, default=0.98, help='momentum for running averages kept by batch norm for test time')
    parser.add_argument('--bn_renorm_rmax', type=float, default=1.5, help='maximum correction scaling done by batch renorm')
    parser.add_argument('--bn_renorm_dmax', type=float, default=0.5, help='maximum correction shift done by batch renorm (in stddev units)')
    parser.add_argument('--bn_renorm_momentum', type=float, default=0.9, help='momentum for running averages kept by batch renorm for training time')
    parser.add_argument('--bn_renorm_shared', type=str2bool, default=False, help='share same set of running statistics for train and test time')
    parser.add_argument('--g_regularization', type=str, default='ortho_cosine', help='[none, ortho, ortho_cosine, l2]')
    parser.add_argument('--g_regularization_factor', type=float, default=0.0001, help='regularization weight for generator')
    parser.add_argument('--conv_padding', type=str, default='reflect', help='[zero / reflect]')
    parser.add_argument('--upsampling_method', type=str, default='deconv4', help='[deconv3 / deconv4 / deconv6 / subpixel2 / subpixel3 / resize_conv]')
    parser.add_argument('--downsampling_method', type=str, default='strided_conv3', help='[strided_conv3 / resize_conv1 / resize_conv3 / pool_only / max_pool_only]')
    parser.add_argument('--g_conv', type=str, default='deconv3', help='[deconv3 / deconv4 / conv3 / conv5]')
    parser.add_argument('--g_grow_factor', type=float, default=2.0, help='channel scale factor for blocks in the generator')
    parser.add_argument('--d_grow_factor', type=float, default=2.0, help='channel scale factor for blocks in the discriminator')
    parser.add_argument('--g_sa_size', type=int, default=0, help='insert generator self-attention after feature map size N, 0 for auto, -1 for no self-attention')
    parser.add_argument('--d_sa_size', type=int, default=0, help='insert discriminator self-attention after feature map size N, 0 for auto, -1 for no self-attention')
    parser.add_argument('--sa_size', type=int, default=0, help='sets both --g_sa_size and --d_sa_size')

    parser.add_argument('--gan_type', type=str, default='ra-dragan', help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge / ra-gan / ra-hinge / ra-dragan / ra-lsgan]')
    parser.add_argument('--d_loss_func', type=str, default='', help='loss function override for the discriminator - if not set, loss function is determined by gan_type')
    parser.add_argument('--activation', type=str, default='prelu', help='default activation function [relu, prelu, lrelu]')
    parser.add_argument('--ld', type=float, default=10.0, help='The gradient penalty lambda')
    parser.add_argument('--multi_head', type=str2bool, default=False, help='enables joint training of two generator heads')

    parser.add_argument('--n_critic', type=int, default=1, help='The number of discriminator updates per generator update')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--sample_num', type=int, default=64, help='The number of sample images')
    parser.add_argument('--static_sample_z', type=str2bool, default=True, help='use the same z vector to generate samples across generations')
    parser.add_argument('--static_sample_seed', type=int, default=123456789, help='seed used to generate z if static z is enabled')
    parser.add_argument('--z_trunc_train', type=str2bool, default=True, help='whether to use truncated normal distribution for z during training')
    parser.add_argument('--z_trunc_sample', type=str2bool, default=True, help='whether to use truncated normal distribution for z during sample generation')
    parser.add_argument('--save_morphs', type=str2bool, default=False, help='when generating samples, also generates a morph between 4 random samples')
    parser.add_argument('--sample_ema', type=str, default='ema', help='[ema, noema, both] whether to generate samples using the moving average of the weights or not')
    parser.add_argument('--random_flip', type=str2bool, default=True, help='if activated, flips input images left or right randomly')

    parser.add_argument('--n_labels', type=int, default=0, help='number of classes or labels')
    parser.add_argument('--cls_embedding', type=str2bool, default=False, help='use a shared embedding for labels')
    parser.add_argument('--cls_embedding_size', type=int, default=0, help='size of class embedding, 0=auto')
    parser.add_argument('--label_file', type=str, default='', help='label file, required if n_labels>0')
    parser.add_argument('--cls_loss_type', type=str, default='logistic', help='loss type for labels [logistic, euclidean]')
    parser.add_argument('--weight_file', type=str, default='', help='weight file containing sample selection probabilities')
    parser.add_argument('--g_first_level_dense_layer', type=str2bool, default=True, help='adds an extra dense layer between the z inputs and the 4x4 layer')
    parser.add_argument('--g_other_level_dense_layer', type=str2bool, default=False, help='adds an extra dense layer between the z inputs and the intermediate layers')
    parser.add_argument('--g_z_dense_concat', type=str2bool, default=False, help='concatenates results of dense layers to z vectors instead of replacing them')
    parser.add_argument('--d_cls_dense_layers', type=str2bool, default=False, help='adds two extra layers before the classification output of the discriminator')
    parser.add_argument('--g_mixed_resblocks', type=str2bool, default=False, help='adds a residual mixed conv layer after each block')
    parser.add_argument('--g_mixed_resblock_ch_div', type=float, default=2.0,help='divisor for the number of channels in the residual mixed conv layers')
    parser.add_argument('--g_final_layer', type=str2bool, default=False, help='adds a z-powered layer in the generator before the RGB output')
    parser.add_argument('--g_final_layer_extra', type=str2bool, default=False, help='adds another convolution before final output, recommended if --g_final_layer_shortcuts is used')
    parser.add_argument('--g_final_layer_extra_bias', type=str2bool, default=False, help='adds bias to extra layer; recommended false, exists mainly for compatibility reasons')
    parser.add_argument('--g_final_kernel', type=int, default=3, help='kernel size for convolutions in final layer')
    parser.add_argument('--g_final_layer_shortcuts', type=str2bool, default=False, help='adds shortcuts between the individual mixed conv layers and the final output')
    parser.add_argument('--g_final_layer_shortcuts_after', type=int, default=0, help='start adding shortcuts after this layer (0=initial max size layer, 1=first mixed conv layer)')
    parser.add_argument('--g_final_mixed_conv', type=str2bool, default=False, help='adds experimental mixed convolutions in the final generator layer')
    parser.add_argument('--g_final_mixed_conv_stacks', type=int, default=2, help='number of mixed convolution stacks, if enabled')
    parser.add_argument('--g_final_mixed_conv_z_layers', type=str, default='none', help='comma-separated list of mixed conv layers (or "all") that should receive their own z values')
    parser.add_argument('--g_final_mixed_nodeconv2', type=str2bool, default=False, help='replaces 2x2 deconvolution in mixed conv layer')
    parser.add_argument('--g_rgb_mix_kernel', type=int, default=3, help='kernel size for final mixing convolution')
    parser.add_argument('--d_cls_loss_weight', type=float, default=5.0, help='factor for classification loss in the descriminator')
    parser.add_argument('--g_cls_loss_weight', type=float, default=1.0, help='factor for classification loss in the generator')
    parser.add_argument('--save_cls_samples', type=str2bool, default=False, help='when generating samples, additionally generates a set of samples for a random class')
    parser.add_argument('--cls_loss_weights', type=str, default='', help='optional file containing whitespace-separated loss factors for each class')


    parser.add_argument('--test_num', type=int, default=10, help='The number of images generated by the test')

    parser.add_argument('--allow_growth', type=str2bool, default=False, help='whether to let Tensorflow dynamically allocate memory')

    parser.add_argument('--checkpoint', type=str, default='',
                        help='load this checkpoint instead of the latest one')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')
    parser.add_argument('--request_dir', type=str, default='request',
                        help='Directory name to handle requests in service mode')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    gpu_options = tf.GPUOptions(allow_growth = args.allow_growth)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)) as sess:

        gan = BigGAN(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            # launch the graph in a session
            gan.train()

            # visualize learned generator
            gan.visualize_results(args.epoch - 1)

            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

        if args.phase == 'service' :
            gan.service()
            print(" [*] Service exit.")

if __name__ == '__main__':
    main()
