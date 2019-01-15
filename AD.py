import numpy as np
import tensorflow as tf
import pdb
import random
import json
from scipy.stats import mode

import DR_discriminator

import data_utils
import plotting
import model
import utils
import eval

from time import time

begin = time()

# --- get settings --- #
# parse command line arguments, or use defaults
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())
# if a settings file is specified, it overrides command line arguments/defaults
if settings['settings_file']: settings = utils.load_settings_from_file(settings)

# --- get data, split --- #
# samples, pdf, labels = data_utils.get_samples_and_labels(settings)

samples, pdf, labels = data_utils.get_data(settings['data'], settings['seq_length'], settings['seq_step'], settings['num_signals'])

# --- save settings, data --- #
print('Ready to run with settings:')
for (k, v) in settings.items(): print(v, '\t', k)
# add the settings to local environment
# WARNING: at this point a lot of variables appear
locals().update(settings)
json.dump(settings, open('./experiments/settings/' + identifier + '.txt', 'w'), indent=0)

class myADclass():
    def __init__(self, epoch, settings=settings, samples=samples, labels=labels):
        self.epoch = epoch
        self.settings = settings
        self.samples = samples
        self.labels = labels
    def ADfunc(self):
        num_samples_t = self.samples.shape[0]
        print('sample_shape:', self.samples.shape[0])
        print('num_samples_t', num_samples_t)

        # -- only discriminate one batch for one time -- #
        D_test = np.empty([num_samples_t, self.settings['seq_length'], 1])
        DL_test = np.empty([num_samples_t, self.settings['seq_length'], 1])
        L_mb = np.empty([num_samples_t, self.settings['seq_length'], 1])
        for batch_idx in range(0, num_samples_t // self.settings['batch_size']):
            start_pos = batch_idx * self.settings['batch_size']
            end_pos = start_pos + self.settings['batch_size']
            T_mb = self.samples[start_pos:end_pos, :, :]
            L_mmb = self.labels[start_pos:end_pos, :, :]
            para_path = './experiments/parameters/' + self.settings['identifier'] + '_' + str(
                self.settings['seq_length']) + '_' + str(self.epoch) + '.npy'
            D_t, L_t = DR_discriminator.dis_trained_model(self.settings, T_mb, para_path)
            D_test[start_pos:end_pos, :, :] = D_t
            DL_test[start_pos:end_pos, :, :] = L_t
            L_mb[start_pos:end_pos, :, :] = L_mmb

        start_pos = (num_samples_t // self.settings['batch_size']) * self.settings['batch_size']
        end_pos = start_pos + self.settings['batch_size']
        size = samples[start_pos:end_pos, :, :].shape[0]
        fill = np.ones([self.settings['batch_size'] - size, samples.shape[1], samples.shape[2]])
        batch = np.concatenate([samples[start_pos:end_pos, :, :], fill], axis=0)
        para_path = './experiments/parameters/' + self.settings['identifier'] + '_' + str(
            self.settings['seq_length']) + '_' + str(self.epoch) + '.npy'
        D_t, L_t = DR_discriminator.dis_trained_model(self.settings, batch, para_path)
        L_mmb = self.labels[start_pos:end_pos, :, :]
        D_test[start_pos:end_pos, :, :] = D_t[:size, :, :]
        DL_test[start_pos:end_pos, :, :] = L_t[:size, :, :]
        L_mb[start_pos:end_pos, :, :] = L_mmb

        # -- use self-defined evaluation functions -- #
        # -- test different tao values for the detection function -- #
        results = np.zeros([12, 5])
        for i in range(2, 8):
            tao = 0.1 * i

            Accu4, Pre4, Rec4, F14, FPR4, D_L4 = DR_discriminator.detection_statistic(D_test, L_mb, tao)
            print('seq_length:', self.settings['seq_length'])
            print('point-wise-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
                  .format(self.epoch, tao, Accu4, Pre4, Rec4, F14, FPR4))
            results[i - 2, :] = [Accu4, Pre4, Rec4, F14, FPR4]

            Accu5, Pre5, Rec5, F15, FPR5 = DR_discriminator.sample_detection(D_test, L_mb, tao)
            print('seq_length:', self.settings['seq_length'])
            print('sample-wise-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
                  .format(self.epoch, tao, Accu5, Pre5, Rec5, F15, FPR5))
            results[i - 2+6, :] = [Accu5, Pre5, Rec5, F15, FPR5]

        return results



if __name__ == "__main__":
    print('Main Starting...')

    Results = np.empty([settings['num_epochs'], 12, 5])

    for epoch in range(settings['num_epochs']):
    # for epoch in range(50, 60):
        ob = myADclass(epoch)
        Results[epoch, :, :] = ob.ADfunc()

    # res_path = './experiments/plots/Results' + '_' + settings['sub_id'] + '_' + str(
    #     settings['seq_length']) + '.npy'
    # np.save(res_path, Results)

    print('Main Terminating...')
    end = time() - begin
    print('Testing terminated | Training time=%d s' % (end))