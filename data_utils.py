import numpy as np
import pandas as pd
import pdb
import re
from time import time
import json
import random

import model

from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal, invgamma, mode
from scipy.special import gamma
from scipy.misc.pilutil import imresize
from functools import partial
from math import ceil

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler


# --- deal with the SWaT data --- #
def swat_gen(seq_length, seq_step, randomize=False):
    """ Load and serialise """
    try:
        train = np.load('./data/swat.npy')
        print('Loaded swat from .npy')
    except IOError:
        print('Failed to load swat data from .npy, loading from csv')
        # read from the csv
        train = np.loadtxt(open('./data/swat.csv', 'r'), delimiter=',')
        # scale samples from 0 to 1
        mm, nn = train.shape
        for i in range(nn-1):
            train[:, i] /= max(train[:, i])
        # scale from -1 to 1
            train[:, i] = 2*train[:, i] - 1
        # save to the npy
        np.save('./data/swat.npy', train)

    # --- data shape --- #
    m, n = train.shape # m=496800, n=52
    # normal data
    # P1 takes 5 hours for stabilisation, kill the unstable data
    ######################################
    # --- choose the generated variable here --- #
    # univariate
    nn = 1
    # multivariate
    # nn = [1, 2, 3, 4]
    # nn = [1, 8, 18, 28]
    samples = train[21600:, nn]
    labels = train[:, n-1]     # the last colummn is label
    ######################################
    num_samples = (samples.shape[0] - seq_length) // seq_step
    ######################################
    try:
        num_signal = samples.shape[1]
    except IndexError:
        print('tuple index out of range')
        num_signal = 1
    #
    aa = np.empty([num_samples, seq_length, num_signal])
    bb = np.empty([num_samples, seq_length, num_signal])

    for j in range(num_samples):
        # aa[j, :, :] = samples[(j * seq_step):(j * seq_step + 7200)]
        # bb[j, :, :] = labels[(j * seq_step):(j * seq_step + 7200)]
        for i in range(num_signal):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + 7200)]
            bb[j, :, i] = labels[(j*seq_step):(j*seq_step+7200)]

    samples = aa[:, 0:7200:100, :]
    labels = bb[:, 0:7200:100, :]

    return samples, labels

def swat_train(seq_length, seq_step, num_signals, randomize=False):
    """ Load and serialise """
    try:
        train = np.load('./data/swat.npy')
        print('Loaded swat from .npy')
    except IOError:
        print('Failed to load swat data from .npy, loading from csv')
        # read from the csv
        train = np.loadtxt(open('./data/swat.csv', 'r'), delimiter=',')
        # scale samples from 0 to 1
        mm, nn = train.shape
        for i in range(nn-1):
            train[:, i] /= max(train[:, i])
        # scale from -1 to 1
            train[:, i] = 2*train[:, i] - 1
        # save to the npy
        np.save('./data/swat.npy', train)

    # --- data shape --- #
    m, n = train.shape # m=496800, n=52

    samples = train[21600:, 0:n-1]
    labels = train[21600:, n-1]    # the last colummn is label
    #############################
    # --- choose variables here --- #
    # samples = samples[:, [1, 2, 3, 4]]
    ############################

    #####################################
    ####################################
    from sklearn.decomposition import PCA
    X_n = samples
    #####################################
    ####################################
    pca = PCA(n_components=num_signals, svd_solver='full')
    pca.fit(X_n)
    ex_var = pca.explained_variance_ratio_
    pc = pca.components_

    # projected values on the principal component
    # T = XP
    T_n = np.matmul(X_n, pc.transpose(1, 0))

    samples = T_n
    # only for one-dimensional
    # samples = T_n.reshape([samples.shape[0], ])
    # samples_a = T_a.reshape([samples_a.shape[0], ])
    ###########################################
    ###########################################
    num_samples = (samples.shape[0] - seq_length) // seq_step
    aa = np.empty([num_samples, seq_length, num_singls])
    bb = np.empty([num_samples, seq_length, 1])


    for j in range(num_samples):
       bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
       # aa[j, :, :] = np.reshape(samples[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
       for i in range(num_singls):
           aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]

    samples = aa
    labels = bb

    return samples, labels


def swat_test(seq_length, seq_step, num_signals, randomize=False):
    """ Load and serialise """
    try:
        test = np.load('./data/swat_a.npy')
        print('Loaded swat_a from .npy')

    except IOError:
        print('Failed to load swat_a data from .npy, loading from csv')
        # read from the csv
        test = np.loadtxt(open('./data/swat_a.csv'), delimiter=',')
        # scale samples from 0 to 1
        mm1, nn1 = test.shape
        for i in range(nn1-1):
            test[:, i] /= max(test[:, i])
        # scale from -1 to 1
            test[:, i] = 2*test[:, i] - 1
        # save to the npy
        np.save('./data/swat_a.npy', test)
    m1, n1 = test.shape # m1=449919, n1=52

    samples_a = test[:, 0:n1-1]
    labels_a = test[:, n1 - 1]
    #############################
    # choose variable here
    # samples_a = samples_a[:, [1, 2, 3, 4]]
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
    num_samples = (samples_a.shape[0] - seq_length) // seq_step
    aa_a = np.empty([num_samples, seq_length, num_signals])
    bb_a = np.empty([num_samples, seq_length, 1])

    for j in range(num_samples):
       bb_a[j, :, :] = np.reshape(labels_a[(j * seq_step):(j * seq_step + seq_length)], [-1,1])
       for i in range(num_signals):
           aa_a[j, :, i] = samples_a[(j * seq_step):(j*seq_step + seq_length), i]

    samples_a = aa_a
    labels_a = bb_a

    return samples_a, labels_a


# ############################ data pre-processing #################################
# --- to do with loading --- #
def get_samples_and_labels(settings):
    """
    Parse settings options to load or generate correct type of data,
    perform test/train split as necessary, and reform into 'samples' and 'labels'
    dictionaries.
    """
    if settings['data_load_from']:
        data_path = './experiments/data/' + settings['data_load_from'] + '.data.npy'
        print('Loading data from', data_path)
        samples, pdf, labels = get_data('load', settings['seq_length'], settings['seq_step'], settings['num_signals'], data_path)
        print('samples_data_uti_shape', samples.shape)
        train, vali, test = samples['train'], samples['vali'], samples['test']
        train_labels, vali_labels, test_labels = labels['train'], labels['vali'], labels['test']
        del samples, labels
    else:
        # generate the data
        data_vars = ['num_samples', 'seq_length', 'num_signals', 'freq_low',
                'freq_high', 'amplitude_low', 'amplitude_high', 'scale',
                'full_mnist']
        data_settings = dict((k, settings[k]) for k in data_vars if k in settings.keys())
        samples, pdf, labels = get_data(settings['data'], settings['seq_length'], settings['seq_step'], settings['num_signals'], data_settings)
        if 'multivariate_mnist' in settings and settings['multivariate_mnist']:
            seq_length = samples.shape[1]
            samples = samples.reshape(-1, int(np.sqrt(seq_length)), int(np.sqrt(seq_length)))
        if 'normalise' in settings and settings['normalise']: # TODO this is a mess, fix
            print(settings['normalise'])
            norm = True
        else:
            norm = False
        if labels is None:
            train, vali, test = split(samples, [0.6, 0.2, 0.2], normalise=norm)
            train_labels, vali_labels, test_labels = None, None, None
        else:
            train, vali, test, labels_list = split(samples, [0.6, 0.2, 0.2], normalise=norm, labels=labels)
            train_labels, vali_labels, test_labels = labels_list

    labels = dict()
    labels['train'], labels['vali'], labels['test'] = train_labels, vali_labels, test_labels

    samples = dict()
    samples['train'], samples['vali'], samples['test'] = train, vali, test

    # futz around with labels
    # TODO refactor cause this is messy
    if 'one_hot' in settings and settings['one_hot'] and not settings['data_load_from']:
        if len(labels['train'].shape) == 1:
            # ASSUME labels go from 0 to max_val inclusive, find max-val
            max_val = int(np.max([labels['train'].max(), labels['test'].max(), labels['vali'].max()]))
            # now we have max_val + 1 dimensions
            print('Setting cond_dim to', max_val + 1, 'from', settings['cond_dim'])
            settings['cond_dim'] = max_val + 1
            print('Setting max_val to 1 from', settings['max_val'])
            settings['max_val'] = 1

            labels_oh = dict()
            for (k, v) in labels.items():
                A = np.zeros(shape=(len(v), settings['cond_dim']))
                A[np.arange(len(v)), (v).astype(int)] = 1
                labels_oh[k] = A
            labels = labels_oh
        else:
            assert settings['max_val'] == 1
            # this is already one-hot!

    if 'predict_labels' in settings and settings['predict_labels']:
        samples, labels = data_utils.make_predict_labels(samples, labels)
        print('Setting cond_dim to 0 from', settings['cond_dim'])
        settings['cond_dim'] = 0

    # update the settings dictionary to update erroneous settings
    # (mostly about the sequence length etc. - it gets set by the data!)
    settings['seq_length'] = samples['train'].shape[1]
    settings['num_samples'] = samples['train'].shape[0] + samples['vali'].shape[0] + samples['test'].shape[0]
    settings['num_signals'] = samples['train'].shape[2]
    settings['num_generated_features'] = samples['train'].shape[2]

    return samples, pdf, labels

def get_data(data_type, seq_length, seq_step, num_signals, data_options=None):
    """
    Helper/wrapper function to get the requested data.
    """
    labels = None
    pdf = None
    if data_type == 'load':
        data_dict = np.load(data_options).item()
        samples = data_dict['samples']
        pdf = data_dict['pdf']
        labels = data_dict['labels']
    elif data_type == 'sine':
         samples = sine_wave(**data_options)
    elif data_type == 'mnist':
        if data_options['full_mnist']:
            samples, labels = mnist()
        else:
            #samples, labels = load_resized_mnist_0_5(14)
            samples, labels = load_resized_mnist(14)       # this is the 0-2 setting
    elif data_type == 'gp_rbf':
        print(data_options)
        samples, pdf = GP(**data_options, kernel='rbf')
    elif data_type == 'linear':
        samples, pdf = linear(**data_options)
    elif data_type == 'swat':
        samples, labels = swat()
    elif data_type == 'swat_gen':
        samples, labels = swat_gen(seq_length, seq_step)
    elif data_type == 'swat_train':
        samples, labels = swat_train(seq_length, seq_step, num_signals)
    elif data_type == 'swat_test':
        samples, labels = swat_test(seq_length, seq_step, num_signals)
    else:
        raise ValueError(data_type)
    print('Generated/loaded', len(samples), 'samples from data-type', data_type)
    return samples, pdf, labels


def get_batch(samples, batch_size, batch_idx, labels=None):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    if labels is None:
        return samples[start_pos:end_pos], None
    else:
        if type(labels) == tuple: # two sets of labels
            assert len(labels) == 2
            return samples[start_pos:end_pos], labels[0][start_pos:end_pos], labels[1][start_pos:end_pos]
        else:
            assert type(labels) == np.ndarray
            return samples[start_pos:end_pos], labels[start_pos:end_pos]

def get_batch_aaa(samples_aaa, batch_size, batch_idx, labels_aaa=None):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    if labels_aaa is None:
        return samples_aaa[start_pos:end_pos], None
    else:
        if type(labels_aaa) == tuple: # two sets of labels
            assert len(labels_aaa) == 2
            return samples_aaa[start_pos:end_pos], labels_aaa[0][start_pos:end_pos], labels_aaa[1][start_pos:end_pos]
        else:
            assert type(labels_aaa) == np.ndarray
            return samples_aaa[start_pos:end_pos], labels_aaa[start_pos:end_pos]

def normalise_data(train, vali, test, low=-1, high=1):
    """ Apply some sort of whitening procedure
    """
    # remember, data is num_samples x seq_length x signals
    # whiten each signal - mean 0, std 1
    mean = np.mean(np.vstack([train, vali]), axis=(0, 1))
    std = np.std(np.vstack([train - mean, vali - mean]), axis=(0, 1))

    normalised_train = (train - mean) / std
    normalised_vali = (vali - mean) / std
    normalised_test = (test - mean) / std
    #    normalised_data = data - np.nanmean(data, axis=(0, 1))
    #    normalised_data /= np.std(data, axis=(0, 1))

    #    # normalise samples to be between -1 and +1
    # normalise just using train and vali
    #    min_val = np.nanmin(np.vstack([train, vali]), axis=(0, 1))
    #    max_val = np.nanmax(np.vstack([train, vali]), axis=(0, 1))
    #
    #    normalised_train = (train - min_val)/(max_val - min_val)
    #    normalised_train = (high - low)*normalised_train + low
    #
    #    normalised_vali = (vali - min_val)/(max_val - min_val)
    #    normalised_vali = (high - low)*normalised_vali + low
    #
    #    normalised_test = (test - min_val)/(max_val - min_val)
    #    normalised_test = (high - low)*normalised_test + low
    return normalised_train, normalised_vali, normalised_test


def scale_data(train, vali, test, scale_range=(-1, 1)):
    signal_length = train.shape[1]
    num_signals = train.shape[2]
    # reshape everything
    train_r = train.reshape(-1, signal_length * num_signals)
    vali_r = vali.reshape(-1, signal_length * num_signals)
    test_r = test.reshape(-1, signal_length * num_signals)
    # fit scaler using train, vali
    scaler = MinMaxScaler(feature_range=scale_range).fit(np.vstack([train_r, vali_r]))
    # scale everything
    scaled_train = scaler.transform(train_r).reshape(-1, signal_length, num_signals)
    scaled_vali = scaler.transform(vali_r).reshape(-1, signal_length, num_signals)
    scaled_test = scaler.transform(test_r).reshape(-1, signal_length, num_signals)
    return scaled_train, scaled_vali, scaled_test


def split(samples, proportions, normalise=False, scale=False, labels=None, random_seed=None):
    """
    Return train/validation/test split.
    """
    if random_seed != None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    assert np.sum(proportions) == 1
    n_total = samples.shape[0]
    n_train = ceil(n_total * proportions[0])
    n_test = ceil(n_total * proportions[2])
    n_vali = n_total - (n_train + n_test)
    # permutation to shuffle the samples
    shuff = np.random.permutation(n_total)
    train_indices = shuff[:n_train]
    vali_indices = shuff[n_train:(n_train + n_vali)]
    test_indices = shuff[(n_train + n_vali):]
    # TODO when we want to scale we can just return the indices
    assert len(set(train_indices).intersection(vali_indices)) == 0
    assert len(set(train_indices).intersection(test_indices)) == 0
    assert len(set(vali_indices).intersection(test_indices)) == 0
    # split up the samples
    train = samples[train_indices]
    vali = samples[vali_indices]
    test = samples[test_indices]
    # apply the same normalisation scheme to all parts of the split
    if normalise:
        if scale: raise ValueError(normalise, scale)  # mutually exclusive
        train, vali, test = normalise_data(train, vali, test)
    elif scale:
        train, vali, test = scale_data(train, vali, test)
    if labels is None:
        return train, vali, test
    else:
        print('Splitting labels...')
        if type(labels) == np.ndarray:
            train_labels = labels[train_indices]
            vali_labels = labels[vali_indices]
            test_labels = labels[test_indices]
            labels_split = [train_labels, vali_labels, test_labels]
        elif type(labels) == dict:
            # more than one set of labels!  (weird case)
            labels_split = dict()
            for (label_name, label_set) in labels.items():
                train_labels = label_set[train_indices]
                vali_labels = label_set[vali_indices]
                test_labels = label_set[test_indices]
                labels_split[label_name] = [train_labels, vali_labels, test_labels]
        else:
            raise ValueError(type(labels))
        return train, vali, test, labels_split


def make_predict_labels(samples, labels):
    """ Given two dictionaries of samples, labels (already normalised, split etc)
        append the labels on as additional signals in the data
    """
    print('Appending label to samples')
    assert not labels is None
    if len(labels['train'].shape) > 1:
        num_labels = labels['train'].shape[1]
    else:
        num_labels = 1
    seq_length = samples['train'].shape[1]
    num_signals = samples['train'].shape[2]
    new_samples = dict()
    new_labels = dict()
    for (k, X) in samples.items():
        num_samples = X.shape[0]
        lab = labels[k]
        # slow code because i am sick and don't want to try to be smart
        new_X = np.zeros(shape=(num_samples, seq_length, num_signals + num_labels))
        for row in range(num_samples):
            new_X[row, :, :] = np.hstack(
                [X[row, :, :], np.array(seq_length * [(2 * lab[row] - 1).reshape(num_labels)])])
        new_samples[k] = new_X
        new_labels[k] = None
    return new_samples, new_labels


# --- specific data-types --- #

def eICU_task(predict_label=False):
    """
    Load the eICU data for the extreme-value prediction task
    """
    path = 'REDACTED'
    data = np.load(path).item()
    # convert it into similar format
    labels = {'train': data['Y_train'], 'vali': data['Y_vali'], 'test': data['Y_test']}
    samples = {'train': data['X_train'], 'vali': data['X_vali'], 'test': data['X_test']}
    # reshape
    for (k, X) in samples.items():
        samples[k] = X.reshape(-1, 16, 4)
    return samples, labels


def mnist(randomize=False):
    """ Load and serialise """
    try:
        train = np.load('./data/mnist_train.npy')
        print('Loaded mnist from .npy')
    except IOError:
        print('Failed to load MNIST data from .npy, loading from csv')
        # read from the csv
        train = np.loadtxt(open('./data/mnist_train.csv', 'r'), delimiter=',')
        # scale samples from 0 to 1
        train[:, 1:] /= 255
        # scale from -1 to 1
        train[:, 1:] = 2 * train[:, 1:] - 1
        # save to the npy
        np.save('./data/mnist_train.npy', train)
    # the first column is labels, kill them
    labels = train[:, 0]
    samples = train[:, 1:]
    if randomize:
        # not needed for GAN experiments...
        print('Applying fixed permutation to mnist digits.')
        fixed_permutation = np.random.permutation(28 * 28)
        samples = train[:, fixed_permutation]
    samples = samples.reshape(-1, 28 * 28, 1)  # add redundant additional signals
    return samples, labels

def load_resized_mnist_0_5(new_size, randomize=False):
    """ Load resised mnist digits from 0 to 5 """

    samples, labels = mnist()
    print('Resizing...')
    samples = samples[np.in1d(labels, [0, 1, 2, 3, 4, 5])]
    labels = labels[np.in1d(labels, [0, 1, 2, 3, 4, 5])]
    if new_size != 28:
        resized_imgs = [imresize(img.reshape([28, 28]), [new_size, new_size], interp='lanczos').ravel()[np.newaxis].T
                        for img in samples]
        resized_imgs = np.array(resized_imgs)
        resized_imgs = resized_imgs.astype(float)
        resized_imgs /= 255.0
        resized_imgs = 2 * resized_imgs - 1
        np.save('./data/resized_mnist_1_5_samples.npy', resized_imgs)
        np.save('./data/resized_mnist_1_5_labels.npy', labels)
        return resized_imgs, labels
    else:
        return samples, labels


def load_resized_mnist(new_size, from_to_digits=(0, 2), randomize=False):
    """ Load resised mnist digits from 0 to 5 """

    samples, labels = mnist()
    print('Resizing...')
    samples = samples[np.in1d(labels, np.arange(from_to_digits[0], from_to_digits[1] + 1))]
    labels = labels[np.in1d(labels, np.arange(from_to_digits[0], from_to_digits[1] + 1))]
    if new_size != 28:
        resized_imgs = [imresize(img.reshape([28, 28]), [new_size, new_size], interp='lanczos').ravel()[np.newaxis].T
                        for img in samples]
        resized_imgs = np.array(resized_imgs)
        resized_imgs = resized_imgs.astype(float)
        resized_imgs /= 255.0
        resized_imgs = 2 * resized_imgs - 1
        np.save('./data/resized_mnist_' + str(from_to_digits[0]) + '_' + str(from_to_digits[1]) + '_5_samples.npy',
                resized_imgs)
        np.save('./data/resized_mnist_' + str(from_to_digits[0]) + '_' + str(from_to_digits[1]) + '_labels.npy', labels)
        return resized_imgs, labels
    else:
        return samples, labels


def sine_wave(seq_length=30, num_samples=28 * 5 * 100, num_signals=1,
              freq_low=1, freq_high=5, amplitude_low=0.1, amplitude_high=0.9, **kwargs):
    ix = np.arange(seq_length) + 1
    samples = []
    for i in range(num_samples):
        signals = []
        for i in range(num_signals):
            f = np.random.uniform(low=freq_high, high=freq_low)  # frequency
            A = np.random.uniform(low=amplitude_high, high=amplitude_low)  # amplitude
            # offset
            offset = np.random.uniform(low=-np.pi, high=np.pi)
            signals.append(A * np.sin(2 * np.pi * f * ix / float(seq_length) + offset))
        samples.append(np.array(signals).T)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples)
    return samples


def periodic_kernel(T, f=1.45 / 30, gamma=7.0, A=0.1):
    """
    Calculates periodic kernel between all pairs of time points (there
    should be seq_length of those), returns the Gram matrix.
    f is frequency - higher means more peaks
    gamma is a scale, smaller makes the covariance peaks shallower (smoother)

    Heuristic for non-singular rbf:
        periodic_kernel(np.arange(len), f=1.0/(0.79*len), A=1.0, gamma=len/4.0)
    """
    dists = squareform(pdist(T.reshape(-1, 1)))
    cov = A * np.exp(-gamma * (np.sin(2 * np.pi * dists * f) ** 2))
    return cov


def GP(seq_length=30, num_samples=28 * 5 * 100, num_signals=1, scale=0.1, kernel='rbf', **kwargs):
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.empty(shape=(num_samples, seq_length, num_signals))
    # T = np.arange(seq_length)/seq_length    # note, between 0 and 1
    T = np.arange(seq_length)  # note, not between 0 and 1
    if kernel == 'periodic':
        cov = periodic_kernel(T)
    elif kernel == 'rbf':
        cov = rbf_kernel(T.reshape(-1, 1), gamma=scale)
    else:
        raise NotImplementedError
    # scale the covariance
    cov *= 0.2
    # define the distribution
    mu = np.zeros(seq_length)
    print(np.linalg.det(cov))
    distribution = multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov)
    pdf = distribution.logpdf
    # now generate samples
    for i in range(num_signals):
        samples[:, :, i] = distribution.rvs(size=num_samples)
    return samples, pdf


def linear_marginal_likelihood(Y, X, a0, b0, mu0, lambda0, log=True, **kwargs):
    """
    Marginal likelihood for linear model.
    See https://en.wikipedia.org/wiki/Bayesian_linear_regression pretty much
    """
    seq_length = Y.shape[1]  # note, y is just a line (one channel) TODO
    n = seq_length
    an = a0 + 0.5 * n
    XtX = np.dot(X.T, X)
    lambdan = XtX + lambda0
    prefactor = (2 * np.pi) ** (-0.5 * n)
    dets = np.sqrt(np.linalg.det(lambda0) / np.linalg.det(lambdan))
    marginals = np.empty(Y.shape[0])
    for (i, y) in enumerate(Y):
        y_reshaped = y.reshape(seq_length)
        betahat = np.dot(np.linalg.inv(XtX), np.dot(X.T, y_reshaped))
        mun = np.dot(np.linalg.inv(lambdan), np.dot(XtX, betahat) + np.dot(lambda0, mu0))
        bn = b0 + 0.5 * (np.dot(y_reshaped.T, y_reshaped) + np.dot(np.dot(mu0.T, lambda0), mu0) - np.dot(
            np.dot(mun.T, lambdan), mun))
        bs = (b0 ** a0) / (bn ** an)
        gammas = gamma(an) / gamma(a0)
        marginals[i] = prefactor * dets * bs * gammas
    if log:
        marginals = np.log(marginals)
    return marginals


def linear(seq_length=30, num_samples=28 * 5 * 100, a0=10, b0=0.01, k=2, **kwargs):
    """
    Generate data from linear trend from probabilistic model.

    The invgamma function in scipy corresponds to wiki defn. of inverse gamma:
        scipy a = wiki alpha = a0
        scipy scale = wiki beta = b0

    k is the number of regression coefficients (just 2 here, slope and intercept)
    """
    T = np.zeros(shape=(seq_length, 2))
    T[:, 0] = np.arange(seq_length)
    T[:, 1] = 1  # equivalent to X
    lambda0 = 0.01 * np.eye(k)  # diagonal covariance for beta
    y = np.zeros(shape=(num_samples, seq_length, 1))
    sigmasq = invgamma.rvs(a=a0, scale=b0, size=num_samples)
    increasing = np.random.choice([-1, 1], num_samples)  # flip slope
    for n in range(num_samples):
        sigmasq_n = sigmasq[n]
        offset = np.random.uniform(low=-0.5, high=0.5)  # todo limits
        mu0 = np.array([increasing[n] * (1.0 - offset) / seq_length, offset])
        beta = multivariate_normal.rvs(mean=mu0, cov=sigmasq_n * lambda0)
        epsilon = np.random.normal(loc=0, scale=np.sqrt(sigmasq_n), size=seq_length)
        y[n, :, :] = (np.dot(T, beta) + epsilon).reshape(seq_length, 1)
    marginal = partial(linear_marginal_likelihood, X=T, a0=a0, b0=b0, mu0=mu0, lambda0=lambda0)
    samples = y
    pdf = marginal
    return samples, pdf


def changepoint_pdf(Y, cov_ms, cov_Ms):
    """
    """
    seq_length = Y.shape[0]
    logpdf = []
    for (i, m) in enumerate(range(int(seq_length / 2), seq_length - 1)):
        Y_m = Y[:m, 0]
        Y_M = Y[m:, 0]
        M = seq_length - m
        # generate mean function for second part
        Ymin = np.min(Y_m)
        initial_val = Y_m[-1]
        if Ymin > 1:
            final_val = (1.0 - M / seq_length) * Ymin
        else:
            final_val = (1.0 + M / seq_length) * Ymin
        mu_M = np.linspace(initial_val, final_val, M)
        # ah yeah
        logpY_m = multivariate_normal.logpdf(Y_m, mean=np.zeros(m), cov=cov_ms[i])
        logpY_M = multivariate_normal.logpdf(Y_M, mean=mu_M, cov=cov_Ms[i])
        logpdf_m = logpY_m + logpY_M
        logpdf.append(logpdf_m)
    return logpdf


def changepoint_cristobal(seq_length=30, num_samples=28 * 5 * 100):
    """
    Porting Cristobal's code for generating data with a changepoint.
    """
    raise NotImplementedError

    basal_values_signal_a = np.random.randn(n_samples) * 0.33
    trends_seed_a = np.random.randn(n_samples) * 0.005
    trends = np.array([i * trends_seed_a for i in range(51)[1:]]).T
    signal_a = (basal_values_signal_a + trends.T).T
    time_noise = np.random.randn(n_samples, n_steps) * 0.01
    signal_a = time_noise + signal_a

    basal_values_signal_b = np.random.randn(n_samples) * 0.33
    trends_seed_b = np.random.randn(n_samples) * 0.005
    trends = np.array([i * trends_seed_b for i in range(51)[1:]]).T
    signal_b = (basal_values_signal_b + trends.T).T
    time_noise = np.random.randn(n_samples, n_steps) * 0.01
    signal_b = time_noise + signal_b

    signal_a = np.clip(signal_a, -1, 1)
    signal_b = np.clip(signal_b, -1, 1)

    # the change in the trend is based on the top extreme values of each
    # signal in the first half
    time_steps_until_change = np.max(np.abs(signal_a), axis=1) + np.max(np.abs(signal_b), axis=1) * 100
    # noise added to the starting point
    time_steps_until_change += np.random.randn(n_samples) * 5
    time_steps_until_change = np.round(time_steps_until_change)
    time_steps_until_change = np.clip(time_steps_until_change, 0, n_steps - 1)
    time_steps_until_change = n_steps - 1 - time_steps_until_change

    trends = np.array([i * trends_seed_a for i in range(101)[51:]]).T
    signal_a_target = (basal_values_signal_a + trends.T).T
    time_noise = np.random.randn(n_samples, n_steps) * 0.01
    signal_a_target = time_noise + signal_a_target

    trends = np.array([i * trends_seed_b for i in range(101)[51:]]).T
    signal_b_target = (basal_values_signal_b + trends.T).T
    time_noise = np.random.randn(n_samples, n_steps) * 0.01
    signal_b_target = time_noise + signal_b_target

    signal_multipliers = []
    for ts in time_steps_until_change:
        signal_multiplier = []
        if ts > 0:
            for i in range(int(ts)):
                signal_multiplier.append(1)
            i += 1
        else:
            i = 0
        multiplier = 1.25
        while (i < n_steps):
            signal_multiplier.append(multiplier)
            multiplier += 0.25
            i += 1
        signal_multipliers.append(signal_multiplier)
    signal_multipliers = np.array(signal_multipliers)

    for s_idx, signal_choice in enumerate(basal_values_signal_b > basal_values_signal_a):
        if signal_choice == False:
            signal_a_target[s_idx] *= signal_multipliers[s_idx]
        else:
            signal_b_target[s_idx] *= signal_multipliers[s_idx]

    signal_a_target = np.clip(signal_a_target, -1, 1)
    signal_b_target = np.clip(signal_b_target, -1, 1)

    # merging signals
    signal_a = np.swapaxes(signal_a[np.newaxis].T, 0, 1)
    signal_b = np.swapaxes(signal_b[np.newaxis].T, 0, 1)
    signal_a_target = np.swapaxes(signal_a_target[np.newaxis].T, 0, 1)
    signal_b_target = np.swapaxes(signal_b_target[np.newaxis].T, 0, 1)
    input_seqs = np.dstack((signal_a, signal_b))
    target_seqs = np.dstack((signal_a_target, signal_b_target))
    return False


def changepoint(seq_length=30, num_samples=28 * 5 * 100):
    """
    Generate data from two GPs, roughly speaking.
    The first part (up to m) is as a normal GP.
    The second part (m to end) has a linear downwards trend conditioned on the
    first part.
    """
    print('Generating samples from changepoint...')
    T = np.arange(seq_length)
    # sample breakpoint from latter half of sequence
    m_s = np.random.choice(np.arange(int(seq_length / 2), seq_length - 1), size=num_samples)
    samples = np.zeros(shape=(num_samples, seq_length, 1))
    # kernel parameters and stuff
    gamma = 5.0 / seq_length
    A = 0.01
    sigmasq = 0.8 * A
    lamb = 0.0  # if non-zero, cov_M risks not being positive semidefinite...
    kernel = partial(rbf_kernel, gamma=gamma)
    # multiple values per m
    N_ms = []
    cov_ms = []
    cov_Ms = []
    pdfs = []
    for m in range(int(seq_length / 2), seq_length - 1):
        # first part
        M = seq_length - m
        T_m = T[:m].reshape(m, 1)
        cov_m = A * kernel(T_m.reshape(-1, 1), T_m.reshape(-1, 1))
        cov_ms.append(cov_m)
        # the second part
        T_M = T[m:].reshape(M, 1)
        cov_mM = kernel(T_M.reshape(-1, 1), T_m.reshape(-1, 1))
        cov_M = sigmasq * (np.eye(M) - lamb * np.dot(np.dot(cov_mM, np.linalg.inv(cov_m)), cov_mM.T))
        cov_Ms.append(cov_M)
    for n in range(num_samples):
        m = m_s[n]
        M = seq_length - m
        # sample the first m
        cov_m = cov_ms[m - int(seq_length / 2)]
        Xm = multivariate_normal.rvs(cov=cov_m)
        # generate mean function for second
        Xmin = np.min(Xm)
        initial_val = Xm[-1]
        if Xmin > 1:
            final_val = (1.0 - M / seq_length) * Xmin
        else:
            final_val = (1.0 + M / seq_length) * Xmin
        mu_M = np.linspace(initial_val, final_val, M)
        # sample the rest
        cov_M = cov_Ms[m - int(seq_length / 2)]
        XM = multivariate_normal.rvs(mean=mu_M, cov=cov_M)
        # combine the sequence
        # NOTE: just one dimension
        samples[n, :, 0] = np.concatenate([Xm, XM])
    pdf = partial(changepoint_pdf, cov_ms=cov_ms, cov_Ms=cov_Ms)
    return samples, pdf, m_s


