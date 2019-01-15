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
import eval

import DR_discriminator

from time import time
from math import floor
import mmd
from mmd import rbf_mmd2, median_pairwise_distance, mix_rbf_mmd2_and_ratio

tf.logging.set_verbosity(tf.logging.ERROR)

# --- get settings --- #
# parse command line arguments, or use defaults
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())
# if a settings file is specified, it overrides command line arguments/defaults
if settings['settings_file']: settings = utils.load_settings_from_file(settings)

# --- get data, split --- #
samples, pdf, labels = data_utils.get_samples_and_labels(settings)

# samples_aaa = np.load('./data/samples_aa.npy')
# labels_aaa = np.load('./data/labels_aa.npy')


# --- save settings, data --- #
print('Ready to run with settings:')
for (k, v) in settings.items(): print(v, '\t', k)
# add the settings to local environment
# WARNING: at this point a lot of variables appear
locals().update(settings)
json.dump(settings, open('./experiments/settings/' + identifier + '.txt', 'w'), indent=0)

if not data == 'load':
    data_path = './experiments/data/' + identifier + '.data.npy'
    np.save(data_path, {'samples': samples, 'pdf': pdf, 'labels': labels})
    print('Saved training data to', data_path)

# --- build model --- #

Z, X, T, CG, CD, CS = model.create_placeholders(batch_size, seq_length, latent_dim, num_signals, cond_dim)

discriminator_vars = ['hidden_units_d', 'seq_length', 'cond_dim', 'batch_size', 'batch_mean']
discriminator_settings = dict((k, settings[k]) for k in discriminator_vars)
generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 'num_generated_features', 'cond_dim', 'learn_scale']
generator_settings = dict((k, settings[k]) for k in generator_vars)

CGAN = (cond_dim > 0)
if CGAN: assert not predict_labels

D_loss, G_loss= model.GAN_loss(Z, X, generator_settings, discriminator_settings,
                                kappa, CGAN, CG, CD, CS, wrong_labels=wrong_labels)
D_solver, G_solver, priv_accountant = model.GAN_solvers(D_loss, G_loss, learning_rate, batch_size,
                                                        total_examples=samples['train'].shape[0],
                                                        l2norm_bound=l2norm_bound,
                                                        batches_per_lot=batches_per_lot, sigma=dp_sigma, dp=dp)

G_sample = model.generator(Z, **generator_settings, reuse=True, c=CG)

D_t, L_t = model.discriminator(T, **discriminator_settings, reuse=True,)
# D_pro = tf.reduce_mean(D_t)
D_pro = D_t
L_pro = L_t
# --- evaluation --- #

# frequency to do visualisations
vis_freq = max(6600 // num_samples, 1)
eval_freq = max(6600// num_samples, 1)

# get heuristic bandwidth for mmd kernel from evaluation samples
heuristic_sigma_training = median_pairwise_distance(samples['vali'])
best_mmd2_so_far = 1000


sess = tf.Session()
sess.run(tf.global_variables_initializer())
# tf.initialize_all_variables().run()

vis_Z = model.sample_Z(batch_size, seq_length, latent_dim, use_time)
T_mb, L_mb = model.sample_T(batch_size)

# T_indices = np.random.choice(len(samples_aaa), size=batch_size)
# T_mb = samples_aaa[T_indices, :, :]
# L_mb = labels_aaa[T_indices, :, :]

# create VARIABLE Z for invert generation
# Zs = tf.get_variable(name='Zs', shape=[batch_size, seq_length, latent_dim], initializer=tf.random_normal_initializer())
# aaa = Zs.shape
# print('Zs:{}'.format(aaa))
# sess.run(tf.global_variables_initializer())
# Z_latent = sess.run(Zs, feed_dict=None)


# generate vis_sample
if CGAN:
    vis_C = model.sample_C(batch_size, cond_dim, max_val, one_hot)
    if 'mnist' in data:
        if one_hot:
            if cond_dim == 6:
                vis_C[:6] = np.eye(6)
            elif cond_dim == 3:
                vis_C[:3] = np.eye(3)
                vis_C[3:6] = np.eye(3)
            else:
                raise ValueError(cond_dim)
        else:
            if cond_dim == 6:
                vis_C[:6] = np.arange(cond_dim)
            elif cond_dim == 3:
                vis_C = np.tile(np.arange(3), 2)
            else:
                raise ValueError(cond_dim)
    elif 'eICU_task' in data:
        vis_C = labels['train'][np.random.choice(labels['train'].shape[0], batch_size, replace=False), :]
    vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, CG: vis_C})
else:
    vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z}) # what is this for? this is the generation process...following is validation process
    vis_C = None

# plot the real data
vis_real_indices = np.random.choice(len(samples['vali']), size=16, replace=False)
vis_real = np.float32(samples['vali'][vis_real_indices, :, :])
if not labels['vali'] is None:
    vis_real_labels = labels['vali'][vis_real_indices]
else:
    vis_real_labels = None
if data == 'mnist':
    if predict_labels:
        assert labels['vali'] is None
        n_labels = 1
        if one_hot:
            n_labels = 6
            lab_votes = np.argmax(vis_real[:, :, -n_labels:], axis=2)
        else:
            lab_votes = vis_real[:, :, -n_labels:]
        labs, _ = mode(lab_votes, axis=1)
        samps = vis_real[:, :, :-n_labels]
    else:
        labs = None
        samps = vis_real
    if multivariate_mnist:
        plotting.save_mnist_plot_sample(samps.reshape(-1, seq_length ** 2, 1), 0, identifier + '_real', n_samples=6,
                                        labels=labs)
    else:
        plotting.save_mnist_plot_sample(samps, 0, identifier + '_real', n_samples=6, labels=labs)
elif 'eICU' in data:
    plotting.vis_eICU_patients_downsampled(vis_real, resample_rate_in_min,
                                           identifier=identifier + '_real', idx=0)
else:
    plotting.save_plot_sample(vis_real, 0, identifier + '_real', n_samples=16,
                              num_epochs=num_epochs)

# for dp
target_eps = [0.125, 0.25, 0.5, 1, 2, 4, 8]
dp_trace = open('./experiments/traces/' + identifier + '.dptrace.txt', 'w')
dp_trace.write('epoch ' + ' eps'.join(map(str, target_eps)) + '\n')

trace = open('./experiments/traces/' + identifier + '.trace.txt', 'w')
trace.write('epoch time D_loss G_loss mmd2 that sample_pdf real_pdf\n')

# --- train --- #
train_vars = ['batch_size', 'D_rounds', 'G_rounds', 'use_time', 'seq_length',
              'latent_dim', 'num_generated_features', 'cond_dim', 'max_val',
              'WGAN_clip', 'one_hot']
train_settings = dict((k, settings[k]) for k in train_vars)

t0 = time()
best_epoch = 0
print('epoch\ttime\tD_loss\tG_loss\tmmd2\tthat\tpdf_sample\tpdf_real')


for epoch in range(num_epochs):
    D_loss_curr, G_loss_curr = model.train_epoch(epoch, samples['train'], labels['train'],
                                                 sess, Z, X, CG, CD, CS,
                                                 D_loss, G_loss,
                                                 D_solver, G_solver, **train_settings)

    # save parameters
    # model.dump_parameters(identifier + '_' + str(epoch), sess)

    # -- eval -- #

    # visualise plots of generated samples, with/without labels
    if epoch % vis_freq == 0: #?

        vis_ZZ = model.sample_Z(batch_size, seq_length, latent_dim, use_time)
        T_mb, L_mb = model.sample_T(batch_size)

        if CGAN:
            vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, CG: vis_C})
        else:
            vis_sample = sess.run(G_sample, feed_dict={Z: vis_ZZ})

        plotting.visualise_at_epoch(vis_sample, data, predict_labels, one_hot, epoch, identifier,
                                   num_epochs, resample_rate_in_min, multivariate_mnist, seq_length, labels=vis_sample)

        # DR_discriminator.save_samples(vis_sample, epoch)

        D_test, L_test = sess.run([D_pro, L_pro], feed_dict={T: T_mb})
        sss = D_test.shape
        print('D_test shape:{}'.format(sss))

        # DR_Pro = 1-tf.reduce_mean(D_test)
        Accu1, Pre1, Rec1, F11, FPR1, D_L = DR_discriminator.detection_statistic(D_test, L_mb, 0.5)
        print('point-wise-Epoch: {}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
              .format(epoch, Accu1, Pre1, Rec1, F11, FPR1))
        DR_discriminator.anomaly_detection_plot(D_test, T_mb, L_mb, D_L, epoch, identifier)

        Accu, Pre, Rec, F1, FPR = DR_discriminator.sample_detection(D_test, L_mb, 0.5)
        print('sample-wise-Epoch: {}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
              .format(epoch, Accu, Pre, Rec1, F1, FPR))

        f = open("./experiments/plots/Measures.txt", "a")
        f.write('--------------------------------------------\n')
        f.write('point-wise-Epoch: {}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}\n'
                .format(epoch, Accu1, Pre1, Rec1, F11, FPR1))
        f.write('sample-wise-Epoch: {}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}\n'
                .format(epoch, Accu, Pre, Rec, F1, FPR))
        f.close()


        ## compute residuals

        # cast samples to float32
        ts_sample = np.float32(T_mb[:, :, :])
        num = ts_sample.shape[0]

        print('Inverting', num, 'samples using model', settings['identifier'], 'at epoch', epoch,)
        e_tolerance = 0.1
        print('until error is below', e_tolerance)

        # get parameters
        # parameters = load_parameters(settings['identifier'] + '_' + str(epoch))
        # assertions
        # assert samples.shape[2] == settings['num_generated_features']


        # create VARIABLE Z for invert generation
        fd = None
        Zs = tf.get_variable(name='Zs', shape=[batch_size, seq_length, latent_dim], initializer=tf.random_normal_initializer())
        aaa = Zs.shape
        print('Zs:{}'.format(aaa))
        sess.run(tf.global_variables_initializer())
        Z_latent = sess.run(Zs, feed_dict=fd)
        # Zs = model.sample_Z(batch_size, seq_length, latent_dim, use_time)
        # create outputs
        gs_sample = sess.run(G_sample, feed_dict={Z: Z_latent})
        gs_sample = np.float32(gs_sample[:, :, :])
        # gs_sample = model.generator(Zs, **generator_settings, reuse=True, c=CG)


        # define loss mmd-based loss
        heuristic_sigma = mmd.median_pairwise_distance_o(ts_sample)  # this is noisy
        print('heuristic_sigma:', heuristic_sigma)
        Kxx, Kxy, Kyy, wts = mmd._mix_rbf_kernel(gs_sample, ts_sample, sigmas=tf.constant(value=heuristic_sigma, shape=(1, 1)))
        similarity_per_sample = tf.diag_part(Kxy)
        reconstruction_error_per_sample = 1 - similarity_per_sample
        similarity = tf.reduce_mean(similarity_per_sample)
        reconstruction_error = 1 - similarity

        # updater
        # from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
        # from differential_privacy.dp_sgd.dp_optimizer import sanitizer
        # from differential_privacy.privacy_accountant.tf import accountant

        # solver = tf.train.AdamOptimizer().minimize(reconstruction_error_per_sample, var_list=[Zs])
        # solver = tf.train.RMSPropOptimizer(learning_rate=500).minimize(reconstruction_error, var_list=[Zs])
        # solver = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(reconstruction_error_per_sample, var_list=Zs)
        solver = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(reconstruction_error_per_sample, var_list=[Zs])
        # solver = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(reconstruction_error, var_list=[Zs])

        # grad_Z = tf.gradients(reconstruction_error_per_sample, Zs)[0]
        # grad_per_Z = tf.norm(grad_Z, axis=(1, 2))
        # grad_norm = tf.reduce_mean(grad_per_Z)
        # solver = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(reconstruction_error, var_list=[Zs])
        print('Finding latent state corresponding to samples...')

        # sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_variables())
        error = sess.run(reconstruction_error, feed_dict=fd)
        # g_n = sess.run(grad_norm, feed_dict=fd)
        # print(g_n)
        i = 0
        max_iter = 10000
        while np.abs(error) > e_tolerance:
            _ = sess.run(solver, feed_dict=fd)
            error = sess.run(reconstruction_error, feed_dict=fd)
            i += 1
            # print(error)
            if i > max_iter:
                break
        Zs = sess.run(Zs, feed_dict=fd)
        error_per_sample = sess.run(reconstruction_error_per_sample, feed_dict=fd)

        print('Z found in', i, 'iterations with final reconstruction error of', error)
        tf.reset_default_graph()



        # Zs, error_per_sample, heuristic_sigma = DR_discriminator.invert(settings, epoch-1, T_mb, g_tolerance=None, e_tolerance=0.1,
        #                                          n_iter=None, max_iter=10000, heuristic_sigma=None)
        # GG = Zs.shape
        # EE = error_per_sample.shape
        #
        # print('invert-Epoch: {}; GG: {}; error_per_sample: {}; heuristic_sigma: {}'
        #       .format(epoch-1, GG, EE, heuristic_sigma))


        # compute mmd2 and, if available, prob density
        if epoch % eval_freq == 0:
            ## how many samples to evaluate with?
            # real_sample_indices = np.random.choice(len(samples['vali']), size=batch_size, replace=False)
            # real_sample = np.float32(samples['vali'][real_sample_indices, :, :])
            #
            # gs_sample = vis_sample
            #
            # ## MMD
            # heuristic_sigma = mmd.median_pairwise_distance_o(real_sample)
            # Kxx, Kxy, Kyy, d = mmd._mix_rbf_kernel(gs_sample, real_sample, sigmas=tf.constant(value=heuristic_sigma, shape=(1, 1)))
            # similarity_per_sample = tf.diag_part(Kxy)
            # similarity = tf.reduce_mean(similarity_per_sample)
            # generation_error = 1 - similarity
            #
            # # n_sigmas = 2
            # # sigma = tf.get_variable(name='sigma', shape=n_sigmas, initializer=tf.constant_initializer(
            # #     value=np.power(heuristic_sigma, np.linspace(-1, 3, num=n_sigmas))))
            # #
            # # _eps = 1e-8
            # # mmd2, ratio = mmd._mmd2_and_ratio(Kxx, Kxy, Kyy, const_diagonal=d, biased=False, min_var_est=_eps)
            #
            # # mmd2, ratio = sess.run(mix_rbf_mmd2_and_ratio(real_sample, gs_sample, sigma, wts=None, biased=True))
            #
            # G_error = sess.run(generation_error)
            # mmd2 = G_roor

            mmd2 = 'NA'
            that_np = 'NA'

            ## prob density (if available)
            if not pdf is None:
                pdf_sample = np.mean(pdf(gs_sample[:, :, 0]))
                pdf_real = np.mean(pdf(real_sample[:, :, 0]))
            else:
                pdf_sample = 'NA'
                pdf_real = 'NA'
        else:
            # report nothing this epoch
            mmd2 = 'NA'
            that = 'NA'
            pdf_sample = 'NA'
            pdf_real = 'NA'

        ## get 'spent privacy'
        if dp:
            spent_eps_deltas = priv_accountant.get_privacy_spent(sess, target_eps=target_eps)
            # get the moments
            deltas = []
            for (spent_eps, spent_delta) in spent_eps_deltas:
                deltas.append(spent_delta)
            dp_trace.write(str(epoch) + ' ' + ' '.join(map(str, deltas)) + '\n')
            if epoch % 10 == 0: dp_trace.flush()

    ## print
    t = time() - t0
    try:
        print('%d\t%.2f\t%.4f\t%.4f\t%.5f\t%.0f\t%.2f\t%.2f' % (
        epoch, t, D_loss_curr, G_loss_curr, mmd2, that_np, pdf_sample, pdf_real))
    except TypeError:  # pdf are missing (format as strings)
        print('%d\t%.2f\t%.4f\t%.4f\t%s\t%s\t %s\t %s' % (
        epoch, t, D_loss_curr, G_loss_curr, mmd2, that_np, pdf_sample, pdf_real))

    ## save trace
    trace.write(' '.join(map(str, [epoch, t, D_loss_curr, G_loss_curr, mmd2, that_np, pdf_sample, pdf_real])) + '\n')
    if epoch % 10 == 0:
        trace.flush()
        plotting.plot_trace(identifier, xmax=num_epochs, dp=dp)

    if shuffle:  # shuffle the training data
        perm = np.random.permutation(samples['train'].shape[0])
        samples['train'] = samples['train'][perm]
        if labels['train'] is not None:
            labels['train'] = labels['train'][perm]

    if epoch % eval_freq == 0:
        model.dump_parameters(identifier + '_' + str(epoch), sess)


# # ----------------generate num_samples samples from the trained model ------------------#
# # ----------------discriminate m samples from the trained model ------------------------#
# # ----------------obtain latent space samples by invert G ------------------------------#
# idx = 50
# num1 = 47508
# num2 = 3720
# gs_samples = model.sample_trained_model(settings, idx, num1, Z_samples=None, C_samples=None)
#
# dis_t, logits_t = model.dis_trained_model(settings, idx, num2, T_samples=None, C_samples=None)
#
# # Zs, error_per_sample, heuristic_sigma = model.invert(settings, idx, samples, g_tolerance=None, e_tolerance=0.1,
# #                                                     n_iter=None, max_iter=10000, heuristic_sigma=None, C_samples=None)
#
# np.save('./experiments/plots/parameters/gs_sample.npy', gs_samples)
# np.save('./experiments/plots/parameters/dis_t.npy', dis_t)

trace.flush()
plotting.plot_trace(identifier, xmax=num_epochs, dp=dp)
model.dump_parameters(identifier + '_' + str(epoch), sess)

## after-the-fact evaluation
# n_test = vali.shape[0]      # using validation set for now TODO
# n_batches_for_test = floor(n_test/batch_size)
# n_test_eval = n_batches_for_test*batch_size
# test_sample = np.empty(shape=(n_test_eval, seq_length, num_signals))
# test_Z = model.sample_Z(n_test_eval, seq_length, latent_dim, use_time)
# for i in range(n_batches_for_test):
#    test_sample[i*batch_size:(i+1)*batch_size, :, :] = sess.run(G_sample, feed_dict={Z: test_Z[i*batch_size:(i+1)*batch_size]})
# test_sample = np.float32(test_sample)
# test_real = np.float32(vali[np.random.choice(n_test, n_test_eval, replace=False), :, :])
## we can only get samples in the size of the batch...
# heuristic_sigma = median_pairwise_distance(test_real, test_sample)
# test_mmd2, that = sess.run(mix_rbf_mmd2_and_ratio(test_real, test_sample, sigmas=heuristic_sigma, biased=False))
##print(test_mmd2, that)
