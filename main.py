from BigGAN import BigGAN
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of BigGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='celebA-HQ', help='[mnist / cifar10 / custom_dataset]')

    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch per gpu')
    parser.add_argument('--virtual_batches', type=int, default=1, help='number of gradient accumulations per step')
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--d_ch', type=int, default=0, help='base channel number per layer in discriminator')

    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freqy')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')

    parser.add_argument('--g_lr', type=float, default=0.00005, help='learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for discriminator')

    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')
    parser.add_argument('--moving_decay', type=float, default=0.999 , help='moving average decay for generator')

    parser.add_argument('--z_dim', type=int, default=256, help='Dimension of noise vector')
    parser.add_argument('--first_split_ratio', type=int, default=3, help='ratio of z values assigned to the first layer')
    parser.add_argument('--z_reconstruct', type=str2bool, default=False, help='train the discriminator to reconstruct z')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')
    parser.add_argument('--bn_in_d', type=str2bool, default=False, help='whether to use BN in the discriminator')
    parser.add_argument('--upsampling_method', type=str, default='deconv_4', help='[deconv_3 / deconv_4 / resize_conv]')
    parser.add_argument('--g_grow_factor', type=float, default=2.0, help='channel scale factor for blocks in the generator')
    parser.add_argument('--d_grow_factor', type=float, default=2.0, help='channel scale factor for blocks in the discriminator')

    parser.add_argument('--gan_type', type=str, default='ra-dragan', help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge / ra-gan / ra-hinge / ra-dragan]')
    parser.add_argument('--d_loss_func', type=str, default='', help='loss function override for the discriminator - if not set, loss function is determined by gan_type')
    parser.add_argument('--ld', type=float, default=10.0, help='The gradient penalty lambda')

    parser.add_argument('--n_critic', type=int, default=1, help='The number of discriminator updates per generator update')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--sample_num', type=int, default=64, help='The number of sample images')
    parser.add_argument('--static_sample_z', type=str2bool, default=True, help='use the same z vector to generate samples across generations')
    parser.add_argument('--static_sample_seed', type=int, default=123456789, help='seed used to generate z if static z is enabled')
    parser.add_argument('--z_trunc_train', type=str2bool, default=True, help='whether to use truncated normal distribution for z during training')
    parser.add_argument('--z_trunc_sample', type=str2bool, default=True, help='whether to use truncated normal distribution for z during sample generation')
    parser.add_argument('--save_morphs', type=str2bool, default=False, help='when generating samples, also generates a morph between 4 random samples')
    parser.add_argument('--sample_ema', type=str, default='ema', help='[ema, noema, both] whether to generate samples using the moving average of the weights or not')

    parser.add_argument('--n_labels', type=int, default=0, help='number of classes or labels')
    parser.add_argument('--label_file', type=str, default='', help='label file, required if n_labels>0')
    parser.add_argument('--g_first_level_dense_layer', type=str2bool, default=True, help='adds an extra dense layer between the z inputs and the 4x4 layer')
    parser.add_argument('--g_final_layer', type=str2bool, default=False, help='adds a z-powered layer in the generator before the RGB output')
    parser.add_argument('--d_cls_loss_weight', type=float, default=5.0, help='factor for classification loss in the descriminator')
    parser.add_argument('--g_cls_loss_weight', type=float, default=1.0, help='factor for classification loss in the generator')
    parser.add_argument('--save_cls_samples', type=str2bool, default=False, help='when generating samples, additionally generates a set of samples for a random class')
    parser.add_argument('--cls_loss_weights', type=str, default='', help='optional file containing whitespace-separated loss factors for each class')


    parser.add_argument('--test_num', type=int, default=10, help='The number of images generated by the test')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

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
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

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

if __name__ == '__main__':
    main()