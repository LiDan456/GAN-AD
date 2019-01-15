import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import model
import mmd
from mod_core_rnn_cell_impl import LSTMCell


def detection_logits(DL_test, L_mb):
    # point-wise detection for one dimension

    aa = DL_test.shape[0]
    bb = DL_test.shape[1]

    DL_test = DL_test.reshape([aa, bb])
    L_mb = L_mb.reshape([aa, bb])
    D_L = np.empty([aa, bb])

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(aa):
        for j in range(bb):
            if DL_test[i, j] == 1:
                # true/negative
                D_L[i, j] = 0
            else:
                # false/positive
                D_L[i, j] = 1

            A = D_L[i, j]
            B = L_mb[i, j]
            if A == 1 and B == 1:
                TP += 1
            elif A == 1 and B == 0:
                FP += 1
            elif A == 0 and B == 0:
                TN += 1
            elif A == 0 and B == 1:
                FN += 1


    cc = (D_L == L_mb)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)
    Accu = float((N / (aa*bb)) * 100)

    # true positive among all the detected positive
    Pre = (100 * TP) / (TP + FP + 1)
    # true positive among all the real positive
    Rec = (100 * TP) / (TP + FN + 1)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate--false alarm rate
    FPR = (100 * FP) / (FP + TN)

    return Accu, Pre, Rec, F1, FPR, D_L

def detection_statistic(D_test, L_mb, tao):
    # point-wise detection for one dimension

    aa = D_test.shape[0]
    bb = D_test.shape[1]

    D_test = D_test.reshape([aa, bb])
    L_mb = L_mb.reshape([aa, bb])
    D_L = np.empty([aa, bb])

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(aa):
        for j in range(bb):
            if D_test[i, j] > tao:
                # true/negative
                D_L[i, j] = 0
            else:
                # false/positive
                D_L[i, j] = 1

            A = D_L[i, j]
            B = L_mb[i, j]
            if A == 1 and B == 1:
                TP += 1
            elif A == 1 and B == 0:
                FP += 1
            elif A == 0 and B == 0:
                TN += 1
            elif A == 0 and B == 1:
                FN += 1


    # D_loss_real = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)), 1)


    cc = (D_L == L_mb)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)
    Accu = float((N / (aa*bb)) * 100)

    # N = np.count_nonzero(labels == 1)
    # P = np.count_nonzero(labels == 0)

    # true positive among all the detected positive
    Pre = (100 * TP) / (TP + FP + 1)
    # true positive among all the real positive
    Rec = (100 * TP) / (TP + FN + 1)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate--false alarm rate
    FPR = (100 * FP) / (FP + TN)

    return Accu, Pre, Rec, F1, FPR, D_L

def detection_statistic_R_D(D_test, Gs, T_mb, L_mb, tao, lam):
    # point-wise detection for one dimension
    # (1-lambda)*R(x)+lambda*D(x)
    # lambda=0.5?
    # D_test, Gs, T_mb, L_mb  are of same size

    R = np.absolute(Gs - T_mb)

    aa = D_test.shape[0]
    bb = D_test.shape[1]

    D_test = D_test.reshape([aa, bb])
    D_test = 1-D_test
    L_mb = L_mb.reshape([aa, bb])
    R = R.reshape([aa, bb])
    for i in range(aa):
        R[i, :] /= np.sum(R[i, :])

    D_L = np.empty([aa, bb])

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(aa):
        for j in range(bb):
            if (1-lam)*R[i, j] + lam*D_test[i, j] > tao:
                # false
                D_L[i, j] = 1
            else:
                # true
                D_L[i, j] = 0

            A = D_L[i, j]
            B = L_mb[i, j]
            if A == 1 and B == 1:
                TP += 1
            elif A == 1 and B == 0:
                FP += 1
            elif A == 0 and B == 0:
                TN += 1
            elif A == 0 and B == 1:
                FN += 1


    cc = (D_L == L_mb)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)
    Accu = float((N / (aa*bb)) * 100)

    # N = np.count_nonzero(labels == 1)
    # P = np.count_nonzero(labels == 0)

    # true positive among all the detected positive
    Pre = (100 * TP) / (TP + FP + 1)
    # true positive among all the real positive
    Rec = (100 * TP) / (TP + FN)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate
    FPR = (100 * FP) / (FP + TN)

    return Accu, Pre, Rec, F1, FPR, D_L


def sample_detection(D_test, L_mb, tao):
    # sample-wise detection for one dimension

    aa = D_test.shape[0]
    bb = D_test.shape[1]

    D_test = D_test.reshape([aa, bb])
    L_mb = L_mb.reshape([aa, bb])
    L = np.sum(L_mb, 1)
    L[L > 0] = 1

    D_L = np.empty([aa, ])

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(aa):
        if np.mean(D_test[i, :]) > tao:
            # true/negative
            D_L[i] = 0
        else:
            # false/positive
            D_L[i] = 1

        A = D_L[i]
        B = L[i]
        if A==1 and B==1:
            TP += 1
        elif A==1 and B==0:
            FP += 1
        elif A==0 and B==0:
            TN += 1
        elif A==0 and B==1:
            FN += 1

    cc = (D_L == L)
    # cc = list(cc)
    N = list(cc).count(True)
    Accu = float((N / (aa)) * 100)

    # N = np.count_nonzero(labels == 1)
    # P = np.count_nonzero(labels == 0)

    # true positive among all the detected positive
    Pre = (100*TP)/(TP+FP+1)
    # true positive among all the real positive
    Rec = (100*TP)/(TP+FN+1)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2*Pre*Rec)/(100*(Pre+Rec+1))
    # False positive rate
    FPR = (100*FP)/(FP+TN)

    return Accu, Pre, Rec, F1, FPR

def sample_detection_R_D(D_test, Gs, T_mb, L_mb, tao, lam):
    # sample-wise detection for one dimension
    # (1-lambda)*R(x)+lambda*D(x)
    # lambda=0.5?
    # D_test, Gs, T_mb, L_mb  are of same size
    R = np.absolute(Gs - T_mb)

    aa = D_test.shape[0]
    bb = D_test.shape[1]

    D_test = D_test.reshape([aa, bb])
    D_test = 1-D_test
    L_mb = L_mb.reshape([aa, bb])
    L = np.sum(L_mb, 1)
    L[L > 0] = 1

    R = R.reshape([aa, bb])
    for i in range(aa):
        R[i, :] /= np.sum(R[i, :])

    D_L = np.empty([aa, ])


    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(aa):
        if (1-lam)*np.mean(R[i, :])+lam*np.mean(D_test[i, :]) > tao:
            # false
            D_L[i] = 1
        else:
            # true
            D_L[i] = 0

        A = D_L[i]
        B = L[i]
        if A==1 and B==1:
            TP += 1
        elif A==1 and B==0:
            FP += 1
        elif A==0 and B==0:
            TN += 1
        elif A==0 and B==1:
            FN += 1

    cc = (D_L == L)
    # cc = list(cc)
    N = list(cc).count(True)
    Accu = float((N / (aa)) * 100)

    # N = np.count_nonzero(labels == 1)
    # P = np.count_nonzero(labels == 0)

    # true positive among all the detected positive
    Pre = (100*TP)/(TP+FP+1)
    # true positive among all the real positive
    Rec = (100*TP)/(TP+FN)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2*Pre*Rec)/(100*(Pre+Rec+1))
    # False positive rate
    FPR = (100*FP)/(FP+TN)

    return Accu, Pre, Rec, F1, FPR


def CUSUM_det(spe_n, spe_a, labels):

    mu = np.mean(spe_n)
    sigma = np.std(spe_n)

    kk = 3*sigma
    H = 15*sigma
    print('H:', H)

    tar = np.mean(spe_a)

    mm = spe_a.shape[0]

    SH = np.empty([mm, ])
    SL = np.empty([mm, ])

    for i in range(mm):
        SH[-1] = 0
        SL[-1] = 0
        SH[i] = max(0, SH[i-1]+spe_a[i]-(tar+kk))
        SL[i] = min(0, SL[i-1]+spe_a[i]-(tar-kk))


    count = np.empty([mm, ])
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(mm):
        A = SH[i]
        B = SL[i]
        AA = H
        BB = -H
        if A <= AA and B >= BB:
            count[i] = 0
        else:
            count[i] = 1

        C = count[i]
        D = labels[i]
        if C == 1 and D == 1:
            TP += 1
        elif C == 1 and D == 0:
            FP += 1
        elif C == 0 and D == 0:
            TN += 1
        elif C == 0 and D == 1:
            FN += 1

    cc = (count == labels)
    # cc = list(cc)
    N = list(cc).count(True)
    Accu = float((N / (mm)) * 100)

    # N = np.count_nonzero(labels == 1)
    # P = np.count_nonzero(labels == 0)

    # true positive among all the detected positive
    Pre = (100 * TP) / (TP + FP + 1)
    # true positive among all the real positive
    Rec = (100 * TP) / (TP + FN)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate
    FPR = (100 * FP) / (FP + TN)

    return Accu, Pre, Rec, F1, FPR


def SPE(X, pc):
    a = X.shape[0]
    b = X.shape[1]

    spe = np.empty([a])
    # Square Prediction Error (square of residual distance)
    #  spe = X'(I-PP')X
    I = np.identity(b, float) - np.matmul(pc.transpose(1, 0), pc)
    # I = np.matmul(I, I)
    for i in range(a):
        x = X[i, :].reshape([51, 1])
        y = np.matmul(x.transpose(1, 0), I)
        spe[i] = np.matmul(y, x)

    return spe



def generator_o(z, hidden_units_g, seq_length, batch_size, num_generated_features, reuse=False, parameters=None, learn_scale=True):
    """
    If parameters are supplied, initialise as such
    """
    # It is important to specify different variable scopes for the LSTM cells.
    with tf.variable_scope("generator_o") as scope:

        W_out_G_initializer = tf.constant_initializer(value=parameters['generator/W_out_G:0'])
        b_out_G_initializer = tf.constant_initializer(value=parameters['generator/b_out_G:0'])
        try:
            scale_out_G_initializer = tf.constant_initializer(value=parameters['generator/scale_out_G:0'])
        except KeyError:
            scale_out_G_initializer = tf.constant_initializer(value=1)
            assert learn_scale
        lstm_initializer = tf.constant_initializer(value=parameters['generator/rnn/lstm_cell/weights:0'])
        bias_start = parameters['generator/rnn/lstm_cell/biases:0']

        W_out_G = tf.get_variable(name='W_out_G', shape=[hidden_units_g, num_generated_features], initializer=W_out_G_initializer)
        b_out_G = tf.get_variable(name='b_out_G', shape=num_generated_features, initializer=b_out_G_initializer)
        scale_out_G = tf.get_variable(name='scale_out_G', shape=1, initializer=scale_out_G_initializer, trainable=False)

        inputs = z

        cell = LSTMCell(num_units=hidden_units_g,
                        state_is_tuple=True,
                        initializer=lstm_initializer,
                        bias_start=bias_start,
                        reuse=reuse)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=[seq_length] * batch_size,
            inputs=inputs)
        rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
        logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G #out put weighted sum
        output_2d = tf.nn.tanh(logits_2d) # logits operation [-1, 1]
        output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
    return output_3d

def discriminator_o(x, hidden_units_d, num_generated_features, reuse=False, parameters=None):
    with tf.variable_scope("discriminator_0") as scope:

        W_out_D_initializer = tf.constant_initializer(value=parameters['discriminator/W_out_D:0'])
        b_out_D_initializer = tf.constant_initializer(value=parameters['discriminator/b_out_D:0'])

        # W_out_D = tf.get_variable(name='W_out_D', shape=[hidden_units_d, num_generated_features],  initializer=W_out_D_initializer)
        # b_out_D = tf.get_variable(name='b_out_D', shape=num_generated_features, initializer=b_out_D_initializer)

        W_out_D = tf.get_variable(name='W_out_D', shape=[hidden_units_d, 1],
                                  initializer=W_out_D_initializer)
        b_out_D = tf.get_variable(name='b_out_D', shape=1, initializer=b_out_D_initializer)


        inputs = x

        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_d, state_is_tuple=True, reuse=reuse)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=inputs)

        # print(rnn_outputs.get_shape())
        # print(W_out_D.get_shape())
        # print(b_out_D.get_shape())

        logits = tf.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D # output weighted sum
        # rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_d])
        # logits = tf.matmul(rnn_outputs_2d, W_out_D) + b_out_D
        # real logits or actual output layer?
        # logit is a function that maps probabilities ([0,1]) to ([-inf,inf]) ?

        output = tf.nn.sigmoid(logits) # y = 1 / (1 + exp(-x)). output activation [0, 1]. Probability??
        # sigmoid output ([0,1]), Probability?

    return output, logits


def invert(settings, epoch, samples, g_tolerance=None, e_tolerance=0.1,
           n_iter=None, max_iter=10000, heuristic_sigma=None):
    """
    Return the latent space points corresponding to a set of a samples
    ( from gradient descent )
    Note: this function is designed for ONE sample generation
    """
    # num_samples = samples.shape[0]
    # cast samples to float32

    samples = np.float32(samples)

    # get the model
    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))



    # print('Inverting', 1, 'samples using model', settings['identifier'], 'at epoch', epoch,)
    # if not g_tolerance is None:
    #     print('until gradient norm is below', g_tolerance)
    # else:
    #     print('until error is below', e_tolerance)


    # get parameters
    parameters = model.load_parameters(settings['identifier'] + '_' + str(epoch))
    # # assertions
    # assert samples.shape[2] == settings['num_generated_features']
    # create VARIABLE Z
    Z = tf.get_variable(name='Z', shape=[1, settings['seq_length'],
                                         settings['latent_dim']],
                        initializer=tf.random_normal_initializer())
    # create outputs

    G_samples = generator_o(Z, settings['hidden_units_g'], settings['seq_length'],
                          1, settings['num_generated_features'],
                          reuse=False, parameters=parameters)
    # generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 'num_generated_features', 'cond_dim', 'learn_scale']
    # generator_settings = dict((k, settings[k]) for k in generator_vars)
    # G_samples = model.generator(Z, **generator_settings, reuse=True)

    fd = None

    # define loss mmd-based loss
    if heuristic_sigma is None:
        heuristic_sigma = mmd.median_pairwise_distance_o(samples)  # this is noisy
        print('heuristic_sigma:', heuristic_sigma)
    samples = tf.reshape(samples, [1, settings['seq_length'], settings['num_generated_features']])
    Kxx, Kxy, Kyy, wts = mmd._mix_rbf_kernel(G_samples, samples, sigmas=tf.constant(value=heuristic_sigma, shape=(1, 1)))
    similarity_per_sample = tf.diag_part(Kxy)
    reconstruction_error_per_sample = 1 - similarity_per_sample
    # reconstruction_error_per_sample = tf.reduce_sum((tf.nn.l2_normalize(G_samples, dim=1) - tf.nn.l2_normalize(samples, dim=1))**2, axis=[1,2])
    similarity = tf.reduce_mean(similarity_per_sample)
    reconstruction_error = 1 - similarity
    # updater
    #    solver = tf.train.AdamOptimizer().minimize(reconstruction_error_per_sample, var_list=[Z])
    # solver = tf.train.RMSPropOptimizer(learning_rate=500).minimize(reconstruction_error, var_list=[Z])
    solver = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(reconstruction_error_per_sample, var_list=[Z])
    # solver = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(reconstruction_error_per_sample, var_list=[Z])

    grad_Z = tf.gradients(reconstruction_error_per_sample, Z)[0]
    grad_per_Z = tf.norm(grad_Z, axis=(1, 2))
    grad_norm = tf.reduce_mean(grad_per_Z)
    # solver = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(reconstruction_error, var_list=[Z])
    print('Finding latent state corresponding to samples...')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        error = sess.run(reconstruction_error, feed_dict=fd)
        g_n = sess.run(grad_norm, feed_dict=fd)
        # print(g_n)
        i = 0
        if not n_iter is None:
            while i < n_iter:
                _ = sess.run(solver, feed_dict=fd)
                error = sess.run(reconstruction_error, feed_dict=fd)
                i += 1
        else:
            if not g_tolerance is None:
                while g_n > g_tolerance:
                    _ = sess.run(solver, feed_dict=fd)
                    error, g_n = sess.run([reconstruction_error, grad_norm], feed_dict=fd)
                    i += 1
                    print(error, g_n)
                    if i > max_iter:
                        break
            else:
                while np.abs(error) > e_tolerance:
                    _ = sess.run(solver, feed_dict=fd)
                    error = sess.run(reconstruction_error, feed_dict=fd)
                    i += 1
                    # print(error)
                    if i > max_iter:
                        break
        Zs = sess.run(Z, feed_dict=fd)
        Gs = sess.run(G_samples, feed_dict={Z: Zs})
        error_per_sample= sess.run(reconstruction_error_per_sample, feed_dict=fd)
        print('Z found in', i, 'iterations with final reconstruction error of', error)
    tf.reset_default_graph()

    return Gs, Zs, error_per_sample, heuristic_sigma


def dis_trained_model(settings, samples, para_path):
    """
    Return the discrimination results of  num_samples testing samples from a trained model described by settings dict
    Note: this function is designed for ONE sample discrimination
    """

    # if settings is a string, assume it's an identifier and load
    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))

    num_samples = samples.shape[0]
    samples = np.float32(samples)
    num_variables = samples.shape[2]
    # samples = np.reshape(samples, [1, settings['seq_length'], settings['num_generated_features']])

    # get the parameters, get other variables
    parameters = model.load_parameters(para_path)
    # create placeholder, T samples

    T = tf.placeholder(tf.float32, [num_samples, settings['seq_length'], num_variables])

    # create the discriminator (GAN or CGAN)
    # normal GAN
    D_t, L_t = discriminator_o(T, settings['hidden_units_d'], num_variables, reuse=False, parameters=parameters)
    # D_t, L_t = model.discriminator(T, settings['hidden_units_d'], settings['seq_length'], num_samples, reuse=False,
    #               parameters=parameters, cond_dim=0, c=None, batch_mean=False)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        D_t, L_t = sess.run([D_t, L_t], feed_dict={T: samples})

    tf.reset_default_graph()
    return D_t, L_t