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

# --- training sample --- #
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

Z, X, T = model.create_placeholders(batch_size, seq_length, latent_dim, num_signals)

discriminator_vars = ['hidden_units_d', 'seq_length', 'batch_size', 'batch_mean']
discriminator_settings = dict((k, settings[k]) for k in discriminator_vars)

generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 'num_generated_features', 'learn_scale']
generator_settings = dict((k, settings[k]) for k in generator_vars)


D_loss, G_loss= model.GAN_loss(Z, X, generator_settings, discriminator_settings)
D_solver, G_solver, priv_accountant = model.GAN_solvers(D_loss, G_loss, learning_rate, batch_size,
                                                        total_examples=samples['train'].shape[0],
                                                        l2norm_bound=l2norm_bound,
                                                        batches_per_lot=batches_per_lot, sigma=dp_sigma, dp=dp)

G_sample = model.generator(Z, **generator_settings, reuse=True)


D_t, L_t = model.discriminator(T, **discriminator_settings, reuse=True)
# D_pro = tf.reduce_mean(D_t)
D_pro = D_t
L_pro = L_t


# --- evaluation settings--- #

# frequency to do visualisations
vis_freq = max(6600 // num_samples, 1)
eval_freq = max(6600 // num_samples, 1)

# get heuristic bandwidth for mmd kernel from evaluation samples
heuristic_sigma_training = median_pairwise_distance(samples['vali'])
best_mmd2_so_far = 1000

# optimise sigma using that (that's t-hat)
batch_multiplier = 5000 // batch_size
eval_size = batch_multiplier * batch_size
eval_eval_size = int(0.2 * eval_size)
eval_real_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
eval_sample_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
n_sigmas = 2
sigma = tf.get_variable(name='sigma', shape=n_sigmas, initializer=tf.constant_initializer(
    value=np.power(heuristic_sigma_training, np.linspace(-1, 3, num=n_sigmas))))
mmd2, that = mix_rbf_mmd2_and_ratio(eval_real_PH, eval_sample_PH, sigma)
with tf.variable_scope("SIGMA_optimizer"):
    sigma_solver = tf.train.RMSPropOptimizer(learning_rate=0.05).minimize(-that, var_list=[sigma])
    # sigma_solver = tf.train.AdamOptimizer().minimize(-that, var_list=[sigma])
    # sigma_solver = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(-that, var_list=[sigma])
sigma_opt_iter = 2000
sigma_opt_thresh = 0.001
sigma_opt_vars = [var for var in tf.global_variables() if 'SIGMA_optimizer' in var.name]


# --- run the program --- #

sess = tf.Session()
sess.run(tf.global_variables_initializer())

vis_Z = model.sample_Z(batch_size, seq_length, latent_dim, use_time)
# T_mb, L_mb = model.sample_T(batch_size, batch_idx)


# Feed in the model inputs of generator and generate samples
# what is this?
vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z})

# plot the real samples
vis_real_indices = np.random.choice(len(samples['vali']), size=16)
vis_real = np.float32(samples['vali'][vis_real_indices, :, :])
if not labels['vali'] is None:
    vis_real_labels = labels['vali'][vis_real_indices]
else:
    vis_real_labels = None
plotting.save_plot_sample(vis_real, 0, identifier + '_real', n_samples=16, num_epochs=num_epochs)


# # for dp # record the running results
# target_eps = [0.125, 0.25, 0.5, 1, 2, 4, 8]
# dp_trace = open('./experiments/traces/' + identifier + '.dptrace.txt', 'w')
# dp_trace.write('epoch ' + ' eps'.join(map(str, target_eps)) + '\n')
#
# trace = open('./experiments/traces/' + identifier + '.trace.txt', 'w')
# trace.write('epoch time D_loss G_loss mmd2 that pdf real_pdf\n')


# --- train --- #
train_vars = ['batch_size', 'D_rounds', 'G_rounds', 'use_time', 'seq_length', 'latent_dim', 'num_generated_features', 'max_val', 'one_hot']
train_settings = dict((k, settings[k]) for k in train_vars)

t0 = time()
best_epoch = 0
print('epoch\ttime\tD_loss\tG_loss\tmmd2\tthat\tpdf_sample\tpdf_real')

# for epoch in range(num_epochs):
for epoch in range(1):
    # -- train epoch -- #
    D_loss_curr, G_loss_curr = model.train_epoch(epoch, samples['train'], labels['train'], sess, Z, X, D_loss, G_loss, D_solver, G_solver, **train_settings)


    # -- eval -- #

    # visualise plots of generated samples, with/without labels
    if epoch % vis_freq == 0: # choose which epoch to visualize
        # prepare for the model inputs
        vis_ZZ = model.sample_Z(batch_size, seq_length, latent_dim, use_time)

        # generate samples for visualization
        vis_sample = sess.run(G_sample, feed_dict={Z: vis_ZZ})
        print('vis_sample shape:{}'.format(vis_sample.shape))
        # plot the generated samples
        plotting.visualise_at_epoch(vis_sample, data, predict_labels, one_hot, epoch, identifier,
                                    num_epochs, resample_rate_in_min, multivariate_mnist, seq_length, labels=vis_sample)

        # plotting.save_samples(vis_sample, epoch)

        # anomaly detection
        # run all the batches
        samples_aa = np.load('./data/samples_aa.npy')
        labels_aa = np.load('./data/labels_aa.npy')
        idx_aa = np.load('./data/idx_aa.npy')

        num_samples_t = samples_aa.shape[0]
        D_test = np.empty([num_samples_t, 120, 1])
        DL_test = np.empty([num_samples_t, 120, 1])
        L_mb = np.empty([num_samples_t, 120, 1])
        I_mb = np.empty([num_samples_t, 120, 1])

        for batch_idx in range(0, num_samples_t//batch_size):
            print('batch_idx:{}'.format(batch_idx))
            start_pos = batch_idx * settings['batch_size']
            end_pos = start_pos + settings['batch_size']
            T_mb = samples_aa[start_pos:end_pos, :, :]
            L_mmb = labels_aa[start_pos:end_pos, :, :]
            I_mmb = idx_aa[start_pos:end_pos, :, :]

            # T_mb, L_mmb, I_mmb = model.sample_T(batch_size, batch_idx)
            D_t, L_t = sess.run([D_pro, L_pro], feed_dict={T: T_mb})
            ss = D_t.shape
            print('D_t shape:{}'.format(ss))
            start_pos = batch_idx * settings['batch_size']
            end_pos = start_pos + settings['batch_size']
            D_test[start_pos:end_pos, :, :] = D_t
            DL_test[start_pos:end_pos, :, :] = L_t
            L_mb[start_pos:end_pos, :, :] = L_mmb
            I_mb[start_pos:end_pos, :, :] = I_mmb

        # T_mb, L_mb, I_mb = model.sample_TT(batch_size)
        # D_test, L_test = sess.run([D_pro, L_pro], feed_dict={T: T_mb})

        sss = D_test.shape
        ssss = L_mb.shape

        print('D_test shape:{}'.format(sss))
        print('L_mb shape:{}'.format(ssss))

        Accu1, Pre1, Rec1, F11, FPR1, D_L = DR_discriminator.detection_logits(DL_test, L_mb)
        print('logits-based-Epoch: {}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
            .format(epoch, Accu1, Pre1, Rec1, F11, FPR1))

        for i in range(3, 8):
            tao = 0.1*i
            Accu1, Pre1, Rec1, F11, FPR1, D_L = DR_discriminator.detection_logits_I(DL_test, L_mb, I_mb, tao)
            print('Comb-logits-based-Epoch: {}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
                .format(epoch, Accu1, Pre1, Rec1, F11, FPR1))

            Accu1, Pre1, Rec1, F11, FPR1, D_L = DR_discriminator.detection_statistic_I(D_test, L_mb, I_mb, tao)
            print('Comb-point-wise-Epoch: {}; tao={}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
                .format(epoch, tao, Accu1, Pre1, Rec1, F11, FPR1))

            Accu1, Pre1, Rec1, F11, FPR1, D_L = DR_discriminator.detection_statistic(D_test, L_mb, tao)
            print('point-wise-Epoch: {}; tao={}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
                  .format(epoch, tao, Accu1, Pre1, Rec1, F11, FPR1))
            # DR_discriminator.anomaly_detection_plot(D_test, T_mb, L_mb, D_L, epoch, identifier)

            Accu, Pre, Rec, F1, FPR = DR_discriminator.sample_detection(D_test, L_mb, tao)
            print('sample-wise-Epoch: {}; tao={}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
                  .format(epoch, tao, Accu, Pre, Rec1, F1, FPR))



        #
        # if data_type == 'swat_mul':
        #     f =  open("./experiments/plots/Measures_P{}.txt".format(settings['process_umber']),"a")
        # else:

        # f = open("./experiments/plots/Measures.txt", "a")
        # f.write('--------------------------------------------\n')
        # f.write('point-wise-Epoch: {}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}\n'
        #         .format(epoch, Accu1, Pre1, Rec1, F11, FPR1))
        # f.write('sample-wise-Epoch: {}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}\n'
        #         .format(epoch, Accu, Pre, Rec, F1, FPR))
        # f.close()


    #compute mmd2 and, if available, prob density
    if epoch % eval_freq == 0:
        ## how many samples to evaluate with?
        eval_Z = model.sample_Z(eval_size, seq_length, latent_dim, use_time)
        eval_sample = np.empty(shape=(eval_size, seq_length, num_signals))
        for i in range(batch_multiplier):
            eval_sample[i * batch_size:(i + 1) * batch_size, :, :] = sess.run(G_sample, feed_dict={ Z: eval_Z[i * batch_size:(i + 1) * batch_size]})
        eval_sample = np.float32(eval_sample)
        eval_real = np.float32(samples['vali'][np.random.choice(len(samples['vali']), size=batch_multiplier * batch_size), :, :])

        eval_eval_real = eval_real[:eval_eval_size]
        eval_test_real = eval_real[eval_eval_size:]
        eval_eval_sample = eval_sample[:eval_eval_size]
        eval_test_sample = eval_sample[eval_eval_size:]

        # MMD
        # reset ADAM variables
        sess.run(tf.initialize_variables(sigma_opt_vars))
        sigma_iter = 0
        that_change = sigma_opt_thresh * 2
        old_that = 0
        while that_change > sigma_opt_thresh and sigma_iter < sigma_opt_iter:
            new_sigma, that_np, _ = sess.run([sigma, that, sigma_solver],
                                             feed_dict={eval_real_PH: eval_eval_real, eval_sample_PH: eval_eval_sample})
            that_change = np.abs(that_np - old_that)
            old_that = that_np
            sigma_iter += 1
        opt_sigma = sess.run(sigma)
        try:
            mmd2, that_np = sess.run(mix_rbf_mmd2_and_ratio(eval_test_real, eval_test_sample, biased=False, sigmas=sigma))
        except ValueError:
            mmd2 = 'NA'
            that = 'NA'

        # save parameters
        if mmd2 < best_mmd2_so_far and epoch > 10:
            best_epoch = epoch
            best_mmd2_so_far = mmd2
        model.dump_parameters(identifier + '_' + str(epoch), sess)

        ## prob density (if available)
        if not pdf is None:
            pdf_sample = np.mean(pdf(eval_sample[:, :, 0]))
            pdf_real = np.mean(pdf(eval_real[:, :, 0]))
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

    model_parameters = dict()
    for v in tf.trainable_variables():
        model_parameters[v.name] = sess.run(v)
    print('P_len:{}'.format(len(model_parameters)))


trace.flush()
plotting.plot_trace(identifier, xmax=num_epochs, dp=dp)
model.dump_parameters(identifier + '_' + str(epoch), sess)

