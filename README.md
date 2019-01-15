# -- Multivariate Anomaly Detection for Time Series Data with GANs -- #

#GAN-AD

This repository contains code for the paper, _[Anomaly Detection with Generative Adversarial Networks for Multivariate Time Series](https://arxiv.org/pdf/1809.04758.pdf)_, by Dan Li, Dacheng Chen, Jonathan Goh, and See-Kiong Ng.

## Overview

We used generative adversarial networks (GANs) to do anomaly detection for time series data.
The GAN framework was **R**GAN that taken from the paper, _[Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs](https://arxiv.org/abs/1706.02633).
Please refer to https://github.com/ratschlab/RGAN for the original code.

## Quickstart

- Python3

- Sample generation

  """python RGAN.py --settings_file gp_gen"""
  
  """python RGAN.py --settings_file sine_gen"""
  
  """python RGAN.py --settings_file mnistfull_gen"""
  
  """python RGAN.py --settings_file swat_gen"""

(Please unpack the mnist_train.7z file in the data folder before generate mnist)

- To train the model for anomaly detection:

  """python RGAN.py --settings_file swat_train"""

- To do anomaly detection:

  """python AD.py --settings_file swat_test"""

## Data

In this repository, we applied GAN-AD on the SWaT dataset, please refer to https://itrust.sutd.edu.sg/ and send request to iTrust is you want to try the data.

