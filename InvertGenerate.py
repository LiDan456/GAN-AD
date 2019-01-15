import tensorflow as tf
import numpy as np
import pdb
import json
from mod_core_rnn_cell_impl import LSTMCell  # modified to allow initializing bias in lstm

import data_utils
import plotting
import model
import mmd
import utils
import eval
import DR_discriminator

from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
from differential_privacy.dp_sgd.dp_optimizer import sanitizer
from differential_privacy.privacy_accountant.tf import accountant



settings = json.load(open('./experiments/settings/swat_test.txt', 'r'))

# T_mb, L_mb = model.sample_T(settings['batch_size'])

# T_samples= np.load('./data/samples_aa.npy')
# T_labels = np.load('./data/labels_aa.npy')

test = np.load('./data/swat_a.npy')
m1, n1 = test.shape
samples_a = test[:, 0:n1-1]
labels_a = test[:, n1 - 1]
############################
# choose variables here
# samples_a = samples_a[:, 1]
############################
############################
from sklearn.decomposition import PCA
#
X_a = samples_a
#####################################
####################################
pca_a = PCA(n_components=num_signals, svd_solver='full')
pca_a.fit(X_a)
pc_a = pca_a.components_

# projected values on the principal component
# T = XP
T_a = np.matmul(X_a, pc_a.transpose(1, 0))

# samples = T_n
samples_a = T_a
# only for one-dimensional
# samples_a = T_a.reshape([samples_a.shape[0], ])
###########################################
###########################################


num_samples = (samples_a.shape[0] - settings['seq_length']) // settings['seq_step']
aa_a = np.empty([num_samples, settings['seq_length'], settings['num_signals']])
bb_a = np.empty([num_samples, settings['seq_length'], 1])

for j in range(num_samples):
    bb_a[j, :, :] = np.reshape(labels_a[(j * settings['seq_step']):(j * settings['seq_step'] + settings['seq_length'])], [-1, 1])
    for i in range(settings['num_signals']):
        aa_a[j, :, i] = samples_a[(j * settings['seq_step']):(j * settings['seq_step'] + settings['seq_length']), i]

# cc_a = aa_a[:, 0:120:10, :]
# dd_a = bb_a[:, 0:120:10, :]

T_samples = aa_a
T_labels = bb_a

############################################################

for epoch in range(55):
    # test_size = 30
    # test_seq = len(T_samples)//30
    # Rs = np.empty(test_seq)

    num = len(T_samples)
    T_index = np.random.choice(num, size=500, replace=False)
    TT_samples = T_samples[T_index, :, :]
    TT_labels = T_labels[T_index, :, :]

    aa = TT_samples.shape[0]
    bb = TT_samples.shape[1]
    cc = TT_samples.shape[2]

    GG = np.empty([aa, bb, cc])
    DD = np.empty([aa, bb, cc])
    for i in range(500):
        # T_mb = T_samples[i*test_size:(i+1)*test_size, :, :]
        # L_mb = T_labels[i*test_size:(i+1)*test_size, :, :]
        T_mb = TT_samples[i, :, :]
        L_mb = TT_labels[i, :, :]

        Gs, Zs, error_per_sample, heuristic_sigma = DR_discriminator.invert(settings, epoch, T_mb, g_tolerance=None,
                                                                        e_tolerance=0.1, n_iter=None, max_iter=10000,
                                                                        heuristic_sigma=None)

        GG[i, :, :] = Gs
        print('sample{}; Gs_shape:{}'.format(i, Gs.shape))

        D_T, L_T = DR_discriminator.dis_trained_model(settings, epoch, T_mb)
        DD[i, :, :] = D_T
        print('sample{}; DT_shape:{}'.format(i, D_T.shape))





    Accu1, Pre1, Rec1, F11, FPR1, D_L = DR_discriminator.detection_statistic_R_D(DD, GG, T_samples, T_labels, 0.5, 0.8)
    print('point-wise-Epoch: {}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
          .format(epoch, Accu1, Pre1, Rec1, F11, FPR1))

    # DR_discriminator.anomaly_detection_plot(D_test, T_mb, L_mb, D_L, epoch, identifier)

    Accu, Pre, Rec, F1, FPR = DR_discriminator.sample_detection_R_D(DD, GG, T_samples, T_labels, 0.5, 0.8)
    print('sample-wise-Epoch: {}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
          .format(epoch, Accu, Pre, Rec1, F1, FPR))

    # record the results here
    # f = open("./experiments/plots/Measures_R_D.txt", "a")
    # f.write('--------------------------------------------\n')
    # f.write('point-wise-Epoch: {}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}\n'
    #         .format(epoch, Accu1, Pre1, Rec1, F11, FPR1))
    # f.write('sample-wise-Epoch: {}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}\n'
    #         .format(epoch, Accu, Pre, Rec, F1, FPR))
    # f.close()