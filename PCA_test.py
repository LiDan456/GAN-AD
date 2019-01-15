import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
import matplotlib.pyplot as plt
import DR_discriminator as dr


##########################################
##########################################
# -- SWaT data -- #
normal = np.load('./data/swat.npy')
anomaly = np.load('./data/swat_a.npy')
# # ALL SENSORS IDX
# # XS = [0, 1, 5, 6, 7, 8, 16, 17, 18, 25, 26, 27, 28, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47]
# # X_n = normal[21600:, XS]
# # X_a = anomaly[:, XS]
# # ALL VARIABLES
X_n = normal[21600:, 0:51]
X_a = anomaly[:, 0:51]
L_a = anomaly[:, 51]
############################################
## -- normalization -- ##
# for i in range(X_a.shape[1]):
#     # print('i=', i)
#     A = max(X_a[:, i])
#     # print('A=', A)
#     if A != 0:
#         X_a[:, i] /= max(X_a[:, i])
#         X_a[:, i] = 2*X_a[:, i] - 1
#     else:
#         X_a[:, i] = X_a[:, i]
# for i in range(X_n.shape[1]):
#     # print('i=', i)
#     B = max(X_n[:, i])
#     # print('B=', B)
#     if B != 0:
#         X_n[:, i] /= max(X_n[:, i])
#         X_n[:, i] = 2*X_n[:, i] - 1
#     else:
#         X_n[:, i] = X_n[:, i]
# print('normalization finished...')
# X = np.concatenate((X_n, X_a), axis=0)
# print(np.isnan(X))
#############################################
pca = PCA(n_components=10, svd_solver='full')
pca.fit(X_n)
ex_var = pca.explained_variance_ratio_
pc = pca.components_
# plt.plot(ex_var, 'b')
# plt.show()

pca_a = PCA(n_components=10, svd_solver='full')
pca_a.fit(X_a)
pc_a = pca_a.components_

# pca_x = PCA(n_components=10, svd_solver='full')
# pca_x.fit(X)
# ex_var_x = pca.explained_variance_ratio_
# pc_x = pca_x.components_
# # plt.plot(ex_var_x, 'b')
# # plt.show()
#############################################
# PC distribution
# A = 10
# pca = PCA(n_components=A, svd_solver='full')
# pca.fit(X_n)
# ex_var = pca.explained_variance_ratio_
# xpoints = range(1, A + 1)
#
# fig, ax = plt.subplots(figsize=(6, 3))
#
# ax.plot(xpoints, ex_var)
# ax.grid(linestyle='--', linewidth=0.5)
# # ax.set_ylim(0, 1)
# ax.set_title('wadi.pca.explained_variance_ratio')
# ax.xaxis.set_ticks(range(1, A + 1, int(A / 10)))
# fig.savefig("./Figs/PC_rate_wadi.png")
# plt.clf()
# plt.close()

# fist four componenta
# pc = pc[0:4, :]
#############################################
def SPE(X, pc):
    a = X.shape[0]
    b = X.shape[1]

    spe = np.empty([a])
    # Square Prediction Error (square of residual distance)
    #  spe = X'(I-PP')X
    I = np.identity(b, float) - np.matmul(pc.transpose(1, 0), pc)
    # I = np.matmul(I, I)
    for i in range(a):
        x = X[i, :].reshape([b, 1])
        y = np.matmul(x.transpose(1, 0), I)
        spe[i] = np.matmul(y, x)

    return spe

spe_n = SPE(X_n, pc)
spe_a = SPE(X_a, pc)

# spe_x = SPE(X, pc_a)

Accu, Pre, Rec, F1, FPR = dr.CUSUM_det(spe_n, spe_a, L_a)
print('SPE_I:Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'.format(Accu, Pre, Rec, F1, FPR))
# f = open("./experiments/plots/Measures_baseline.txt", "a")
# f.write('PCA-SPE:Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}\n'.format(Accu, Pre, Rec, F1, FPR))
# f.close()

#
# # projected values on the principal component
# # T = XP
# T_n = np.matmul(X_n, pc.transpose(1, 0))
# T_a = np.matmul(X_a, pc.transpose(1, 0))
# T_x = np.matmul(X, pc_x.transpose(1, 0))
#
# # projected values
# plt.plot(T_n[0:14400:100], 'b')
# plt.plot(T_a[0:14400:100], 'r')
# # plt.plot(T_x[496800:496800+14400:100], 'g')
# plt.plot(L_a[0:14400:100], 'k')
# plt.show()
#
# # SPE values
# plt.plot(spe_n[0:14400:100], 'b')
# plt.plot(spe_a[0:14400:100], 'r')
# # plt.plot(spe_x[496800:496800+14400:100], 'g')
# plt.plot(30*L_a[0:14400:100], 'k')
# plt.show()
