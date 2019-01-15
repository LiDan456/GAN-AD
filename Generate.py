import numpy as np
import tensorflow as tf
import pdb
import random
import json
from scipy.stats import mode

import data_utils
import plotting
import model
import utils

from time import time
from math import floor

tf.logging.set_verbosity(tf.logging.ERROR)


begin = time()
# --- get settings --- #
# parse command line arguments, or use defaults
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())
# if a settings file is specified, it overrides command line arguments/defaults
if settings['settings_file']: settings = utils.load_settings_from_file(settings)

# --- get data, split --- #
samples, pdf, labels = data_utils.get_samples_and_labels(settings)

# samples, pdf, labels = data_utils.get_data(settings)
# print('samples_shape', samples['vali'].shape)
# --- training sample --- #
# --- save settings, data --- #
print('Ready to run with settings:')
for (k, v) in settings.items(): print(v, '\t', k)
# add the settings to local environment
# WARNING: at this point a lot of variables appear
locals().update(settings)
json.dump(settings, open('./experiments/settings/' + identifier + '.txt', 'w'), indent=0)

# --- build model --- #
# preparation: data placeholders and model parameters
Z, X, T = model.create_placeholders(batch_size, seq_length, latent_dim, num_signals)

discriminator_vars = ['hidden_units_d', 'seq_length', 'batch_size', 'batch_mean']
discriminator_settings = dict((k, settings[k]) for k in discriminator_vars)

generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 'num_generated_features', 'learn_scale']
generator_settings = dict((k, settings[k]) for k in generator_vars)

# model: GAN losses
D_loss, G_loss= model.GAN_loss(Z, X, generator_settings, discriminator_settings)
D_solver, G_solver, priv_accountant = model.GAN_solvers(D_loss, G_loss, learning_rate, batch_size,
                                                        total_examples=samples['train'].shape[0],
                                                        l2norm_bound=l2norm_bound,
                                                        batches_per_lot=batches_per_lot, sigma=dp_sigma, dp=dp)
# model: generate samples for visualization
G_sample = model.generator(Z, **generator_settings, reuse=True)


# --- run the program --- #
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# sess = tf.Session()
sess.run(tf.global_variables_initializer())


# plot the real samples for comparison
vis_real_indices = np.random.choice(len(samples['vali']), size=6, replace=False)
vis_real = np.float32(samples['vali'][vis_real_indices, :, :])
plotting.visualise_at_epoch(vis_real, data, predict_labels, one_hot, 0, identifier+ '_real',
                                num_epochs, resample_rate_in_min, multivariate_mnist, seq_length, labels=None)



# --- train --- #
train_vars = ['batch_size', 'D_rounds', 'G_rounds', 'use_time', 'seq_length', 'latent_dim', 'num_generated_features', 'max_val', 'one_hot']
train_settings = dict((k, settings[k]) for k in train_vars)

t0 = time()
# for epoch in range(num_epochs):
for epoch in range(1):
    # -- train epoch -- #
    D_loss_curr, G_loss_curr = model.train_epoch(epoch, samples['train'], labels['train'], sess, Z, X, D_loss, G_loss, D_solver, G_solver, **train_settings)


    # -- eval -- #

    # visualise plots of generated samples, with/without labels
    # prepare for the model inputs
    vis_ZZ = model.sample_Z(batch_size, seq_length, latent_dim, use_time)

    # generate samples for visualization
    vis_sample = sess.run(G_sample, feed_dict={Z: vis_ZZ})
    print('vis_sample shape:{}'.format(vis_sample.shape))
    # plot the generated samples
    plotting.visualise_at_epoch(vis_sample, data, predict_labels, one_hot, epoch, identifier,
                                num_epochs, resample_rate_in_min, multivariate_mnist, seq_length, labels=None)

    # save the generated samples in cased they might be useful for comparison
    plotting.save_samples(vis_sample, settings['identifier'], epoch)
    t = time() - t0
    print('Epoch:{} | Plotted {} gs_samples | Saved {} gs_samples | Time:{}.'.format(epoch, 6, vis_sample.shape[0], t))

end = time() - begin
# print('Training terminated | Training time=%ds' %(end) )
print("Training terminated | training time = %ds  " % (time() - begin))