import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA

# ################### Raw Data ######################
normal = np.load('./Figs/data/swat.npy')
anomaly = np.load('./Figs/data/swat_a.npy')

# lit101 = 1
# p101 = 3
# ait202 = 6
# dpit301 = 16
# lit301 = 18
# mv303 = 21
# fit401 = 27
# lit401 = 28
# ait504 = 37
#
A = [1, 3, 6, 16, 18, 21, 27, 28, 37]
#
# X_n = normal[21600:, [1, 3, 6, 16, 18, 21, 27, 28, 37]]
# X_a = anomaly[:, [1, 3, 6, 16, 18, 21, 27, 28, 37]]
X_n = normal[21600:, :51]
X_a = anomaly[:, :51]
L_a = anomaly[:, 51]
# #######################G_samples#########################
X_a1 = X_a[4920 - 1200: 5302 + 1000, 1]
X_a1 = X_a1[0:len(X_a1):125]
gs1 = np.load('./Figs/data/swat_gs_lit101_1_57.npy')
X_a2 = X_a[117000 - 300: 117720 + 1800, 1]
X_a2 = X_a2[0:len(X_a2):145]
gs2 = np.load('./Figs/data/swat_gs_lit101_2_52.npy')
X_a3 = X_a[0:7200, 18]
X_a3 = X_a3[0:7200:200]
gs3 = np.load('./Figs/data/swat_gs_lit301_50.npy')
X_a4 = X_n[2000:9200, 1]
X_a4 = X_a4[0:7200:200]
gs4 = np.load('./Figs/data/swat_gs_real_gs1_lit101_92.npy')

X_m5 = X_n[300:7500, [1, 8, 18, 28]]
X_a5 = np.empty([36, 4])
for i in range(4):
    X_a5[:, i] = X_m5[0:7200:200, i]
gs5 = np.load('./Figs/data/swat_gs_real_gs4_82.npy')

X_m6 = X_n[1000:8200, [1, 8, 18, 28]]
X_a6 = np.empty([36, 4])
for i in range(4):
    X_a6[:, i] = X_m6[0:7200:200, i]
gs6 = np.load('./Figs/data/swat_gs_real_gs4_86.npy')

X_m7 = X_n[200:7400, [1, 8, 18, 28]]
X_a7 = np.empty([36, 4])
for i in range(4):
    X_a7[:, i] = X_m7[0:7200:200, i]
gs7 = np.load('./Figs/data/swat_gs_real_gs4_88.npy')

X_m8 = X_n[2000:9200, [1, 8, 18, 28]]
X_a8 = np.empty([36, 4])
for i in range(4):
    X_a8[:, i] = X_m8[0:7200:200, i]
gs8 = np.load('./Figs/data/swat_gs_real_gs4_98.npy')
# ####################### MMD #############################

mmd_0 = np.load('./Figs/data/MMD_gs1_lit101.npy')
mmd_1 = np.load('./Figs/data/MMD_gs4.npy')

# #########################################################

def GS_plot(X_a1, X_a2, X_a3, X_a4, X_a6, X_a5, X_a7, X_a8, gs1, gs2, gs3, gs4, gs5, gs6, gs7, gs8):
    nrow = 4
    ncol = 4
    fig, ax = plt.subplots(nrow, ncol, figsize=(20, 15))
    ax[0, 0].plot(gs1[1, :, :], 'r')
    ax[0, 0].set_title("Generated samples", color='C0')
    ax[0, 1].plot(X_a1, 'k')
    ax[0, 1].set_title("Real samples", color='C0')
    ax[1, 0].plot(gs2[1, :, :], 'r')
    # ax[1, 0].set_title("Generated samples", color='C0')
    ax[1, 1].plot(X_a2, 'k')
    # ax[1, 1].set_title("Real samples", color='C0')
    ax[2, 0].plot(gs3[1, :, :], 'r')
    # ax[2, 0].set_title("Generated samples", color='C0')
    ax[2, 1].plot(X_a3, 'k')
    # ax[2, 1].set_title("Real samples", color='C0')
    ax[3, 0].plot(gs4[1, :, :], 'r')
    # ax[3, 0].set_title("Generated samples", color='C0')
    ax[3, 1].plot(X_a4, 'k')
    # ax[3, 1].set_title("Real samples", color='C0')

    ax[0, 2].plot(gs5[1, :, :], 'r')
    ax[0, 2].set_title("Generated samples", color='C0')
    ax[0, 3].plot(X_a6, 'k')
    ax[0, 3].set_title("Real samples", color='C0')
    ax[1, 2].plot(gs6[1, :, :], 'r')
    # ax[1, 2].set_title("Generated samples", color='C0')
    ax[1, 3].plot(X_a5, 'k')
    # ax[1, 3].set_title("Real samples", color='C0')
    ax[2, 2].plot(gs7[1, :, :], 'r')
    # ax[2, 2].set_title("Generated samples", color='C0')
    ax[2, 3].plot(X_a7, 'k')
    # ax[2, 3].set_title("Real samples", color='C0')
    ax[3, 2].plot(gs8[1, :, :], 'r')
    # ax[3, 2].set_title("Generated samples", color='C0')
    ax[3, 3].plot(X_a8, 'k')
    # ax[3, 3].set_title("Real samples", color='C0')

    fig.subplots_adjust(hspace=0.20)
    fig.savefig("./Figs/GS_RS.png")
    plt.clf()
    plt.close()
    return True

def GS_plotI(X_a1, X_a2, X_a3, X_a4, gs1, gs2, gs3, gs4):
    nrow = 4
    ncol = 2
    fig, ax = plt.subplots(nrow, ncol, figsize=(6, 6))
    ax[0, 0].plot(gs1[1, :, :], 'k--')
    ax[0, 0].set_title("Generated samples", color='C0')
    ax[0, 1].plot(X_a1, 'k')
    ax[0, 1].set_title("Real samples", color='C0')
    ax[1, 0].plot(gs2[1, :, :], 'k--')
    # ax[1, 0].set_title("Generated samples", color='C0')
    ax[1, 1].plot(X_a2, 'k')
    # ax[1, 1].set_title("Real samples", color='C0')
    ax[2, 0].plot(gs3[1, :, :], 'k--')
    # ax[2, 0].set_title("Generated samples", color='C0')
    ax[2, 1].plot(X_a3, 'k')
    # ax[2, 1].set_title("Real samples", color='C0')
    ax[3, 0].plot(gs4[1, :, :], 'k--')
    # ax[3, 0].set_title("Generated samples", color='C0')
    ax[3, 1].plot(X_a4, 'k')
    # ax[3, 1].set_title("Real samples", color='C0')

    fig.subplots_adjust(hspace=0.20)
    fig.savefig("./Figs/GS_RSI.png", bbox_inches='tight')
    plt.clf()
    plt.close()
    return True

def GS_plotII(X_a6, X_a5, X_a7, X_a8, gs5, gs6, gs7, gs8):
    nrow = 4
    ncol = 2
    fig, ax = plt.subplots(nrow, ncol, figsize=(6, 6))
    ax[0, 0].plot(gs5[1, :, :], '--')
    ax[0, 0].set_title("Generated samples", color='C0')
    ax[0, 1].plot(X_a6)
    ax[0, 1].set_title("Real samples", color='C0')
    ax[1, 0].plot(gs6[1, :, :], '--')
    # mpl.rcParams['lines.dashed_pattern'] = [3, 5, 1, 5]
    # ax[1, 0].set_title("Generated samples", color='C0')
    ax[1, 1].plot(X_a5)
    # ax[1, 1].set_title("Real samples", color='C0')
    ax[2, 0].plot(gs7[1, :, :], '--')
    # ax[2, 0].set_title("Generated samples", color='C0')
    ax[2, 1].plot(X_a7)
    # ax[2, 1].set_title("Real samples", color='C0')
    ax[3, 0].plot(gs8[1, :, :], '--')
    # ax[3, 0].set_title("Generated samples", color='C0')
    ax[3, 1].plot(X_a8)
    # ax[3, 1].set_title("Real samples", color='C0')

    fig.subplots_adjust(hspace=0.20)
    fig.savefig("./Figs/GS_RSII.png", bbox_inches='tight')
    plt.clf()
    plt.close()
    return True

def Raw_plot(X_n, X_a, L_a, A):
    X_nor = X_n[0:A, [1, 3, 6, 16, 18, 21, 27, 28, 37]]
    X_att = X_a[0:A, [1, 3, 6, 16, 18, 21, 27, 28, 37]]
    # L_att = L_a[0:A]

    x_points = np.arange(A)
    nrow = 2
    ncol = 9
    fig, ax = plt.subplots(nrow, ncol, figsize=(35, 10))
    for m in range(nrow):
        if m==0:
            for n in range(ncol):
                # first row
                sample = X_nor[:, n]
                ax[m, n].plot(x_points, sample, 'b')
                # ax[m, n].set_ylim(-1, 1)
                if n==0:
                    ax[m, n].set_title("LIT-101_nor", color='C0')
                elif n==1:
                    ax[m, n].set_title("P-101_nor", color='C0')
                elif n==2:
                    ax[m, n].set_title("AIT-202_nor", color='C0')
                elif n==3:
                    ax[m, n].set_title("DPIT-301_nor", color='C0')
                elif n==4:
                    ax[m, n].set_title("LIT-301_nor", color='C0')
                elif n==5:
                    ax[m, n].set_title("MV-303_nor", color='C0')
                elif n==6:
                    ax[m, n].set_title("FIT-401_nor", color='C0')
                elif n==7:
                    ax[m, n].set_title("LIT-401_nor", color='C0')
                elif n==8:
                    ax[m, n].set_title("AIT-504_nor", color='C0')
        else:
            for n in range(ncol):
                # second row
                sample = X_att[:, n]
                # label = L_att[:]
                ax[m, n].plot(x_points, sample, 'r')
                # ax[m, n].plot(x_points, label, 'k')
                # ax[m, n].set_ylim(-1, 1)
                if n==0:
                    ax[m, n].set_title("LIT-101_att", color='C0')
                elif n==1:
                    ax[m, n].set_title("P-101_att", color='C0')
                elif n==2:
                    ax[m, n].set_title("AIT-202_att", color='C0')
                elif n==3:
                    ax[m, n].set_title("DPIT-301_att", color='C0')
                elif n==4:
                    ax[m, n].set_title("LIT-301_att", color='C0')
                elif n==5:
                    ax[m, n].set_title("MV-303_att", color='C0')
                elif n==6:
                    ax[m, n].set_title("FIT-401_att", color='C0')
                elif n==7:
                    ax[m, n].set_title("LIT-401_att", color='C0')
                elif n==8:
                    ax[m, n].set_title("AIT-504_att", color='C0')

    fig.subplots_adjust(hspace=0.15)
    fig.savefig("./Figs/RawData.png")
    plt.clf()
    plt.close()
    return True

def Raw_plot_sensorI(X_n, X_a, L_a, A):
    X_nor = X_n[0:A, [1, 18, 28]]
    X_att = X_a[0:A, [1, 18, 28]]
    # L_att = L_a[0:A]

    x_points = np.arange(A)
    nrow = 2
    ncol = 3
    fig, ax = plt.subplots(nrow, ncol, figsize=(9, 6))
    for m in range(nrow):
        if m==0:
            for n in range(ncol):
                # first row
                sample = X_nor[:, n]
                ax[m, n].plot(x_points, sample, 'b')
                # ax[m, n].set_ylim(-1, 1)
                if n==0:
                    ax[m, n].set_title("LIT-101_nor", color='C0')
                elif n==1:
                    ax[m, n].set_title("LIT-301_nor", color='C0')
                elif n==2:
                    ax[m, n].set_title("LIT-401_nor", color='C0')
        else:
            for n in range(ncol):
                # second row
                sample = X_att[:, n]
                # label = L_att[:]
                ax[m, n].plot(x_points, sample, 'r')
                # ax[m, n].plot(x_points, label, 'k')
                # ax[m, n].set_ylim(-1, 1)
                if n==0:
                    ax[m, n].set_title("LIT-101_att", color='C0')
                elif n==1:
                    ax[m, n].set_title("LIT-301_att", color='C0')
                elif n==2:
                    ax[m, n].set_title("LIT-401_att", color='C0')

    fig.subplots_adjust(hspace=0.3)
    fig.savefig("./Figs/RawData_sensorI.png", bbox_inches='tight')
    plt.clf()
    plt.close()
    return True

def Raw_plot_sensorII(X_n, X_a, L_a, A):
    X_nor = X_n[0:A, [6, 16, 37]]
    X_att = X_a[0:A, [6, 16, 37]]
    # L_att = L_a[0:A]

    x_points = np.arange(A)
    nrow = 2
    ncol = 3
    fig, ax = plt.subplots(nrow, ncol, figsize=(9, 6))
    for m in range(nrow):
        if m==0:
            for n in range(ncol):
                # first row
                sample = X_nor[:, n]
                ax[m, n].plot(x_points, sample, 'b')
                # ax[m, n].set_ylim(-1, 1)
                if n==0:
                    ax[m, n].set_title("AIT-202_nor", color='C0')
                elif n==1:
                    ax[m, n].set_title("DPIT-301_nor", color='C0')
                elif n==2:
                    ax[m, n].set_title("AIT-504_nor", color='C0')
        else:
            for n in range(ncol):
                # second row
                sample = X_att[:, n]
                # label = L_att[:]
                ax[m, n].plot(x_points, sample, 'r')
                # ax[m, n].plot(x_points, label, 'k')
                # ax[m, n].set_ylim(-1, 1)
                if n==0:
                    ax[m, n].set_title("AIT-202_att", color='C0')
                elif n==1:
                    ax[m, n].set_title("DPIT-301_att", color='C0')
                elif n==2:
                    ax[m, n].set_title("AIT-504_att", color='C0')

    fig.subplots_adjust(hspace=0.3)
    fig.savefig("./Figs/RawData_sensorII.png", bbox_inches='tight')
    plt.clf()
    plt.close()
    return True

def Raw_plot_actuator(X_n, X_a, L_a, A):
    X_nor = X_n[100000:100000+A, [3, 21, 0]]
    X_att = X_a[100000:100000+A, [3, 21, 0]]
    # L_att = L_a[0:A]

    x_points = np.arange(A)
    nrow = 2
    ncol = 3
    fig, ax = plt.subplots(nrow, ncol, figsize=(9, 6))
    for m in range(nrow):
        if m==0:
            for n in range(ncol):
                # first row
                sample = X_nor[:, n]
                ax[m, n].plot(x_points, sample, 'b')
                # ax[m, n].set_ylim(-1, 1)
                if n==0:
                    ax[m, n].set_title("P-101_nor", color='C0')
                elif n==1:
                    ax[m, n].set_title("MV-303_nor", color='C0')
                elif n==2:
                    ax[m, n].set_title("FIT-401_nor", color='C0')
        else:
            for n in range(ncol):
                # second row
                sample = X_att[:, n]
                # label = L_att[:]
                ax[m, n].plot(x_points, sample, 'r')
                # ax[m, n].plot(x_points, label, 'k')
                # ax[m, n].set_ylim(-1, 1)
                if n==0:
                    ax[m, n].set_title("P-101_att", color='C0')
                elif n==1:
                    ax[m, n].set_title("MV-303_att", color='C0')
                elif n==2:
                    ax[m, n].set_title("FIT-401_att", color='C0')

    # for n in range(ncol):
    #     ax[-1, n].xaxis.set_ticks(range(0, A, int(A/10)))
    # fig.suptitle()
    fig.subplots_adjust(hspace=0.3)
    fig.savefig("./Figs/RawData_actuator.png", bbox_inches='tight')
    plt.clf()
    plt.close()
    return True

def PC_Rate_plot(X_n, X_a, L_a, A):
    pca = PCA(n_components=A, svd_solver='full')
    pca.fit(X_n)
    ex_var = pca.explained_variance_ratio_
    xpoints = range(1, A + 1)

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(xpoints, ex_var)
    ax.grid(linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_title("pca.explained_variance_ratio", color='C0')
    ax.xaxis.set_ticks(range(1, A+1, int(A/10)))
    fig.savefig("./Figs/PC_rate.png")
    plt.clf()
    plt.close()
    return True

def MMD_plot(mmd_0, mmd_1):
    xpoints = range(100)
    # xpoints = [i for i in xpoints if i%2 == 0]

    fig, ax = plt.subplots(figsize=(6, 3))

    m0, =ax.plot(xpoints, mmd_0, label='Univariate MMD')
    m1, =ax.plot(xpoints, mmd_1, label='Multivariate MMD')
    ax.legend(handles=[m0, m1], labels=['Univariate MMD', 'Multivariate MMD'])
    ax.grid(linestyle='--', linewidth=0.5)
    # ax.set_ylabel('MMD')
    ax.set_title('MMD')
    fig.savefig("./Figs/MMD.png")
    plt.clf()
    plt.close()

    return True

# A = 100000
# Raw_plot(X_n, X_a, L_a, A)
# Raw_plot_actuator(X_n, X_a, L_a, A)
# Raw_plot_sensorI(X_n, X_a, L_a, A)
# Raw_plot_sensorII(X_n, X_a, L_a, A)

# PC_Rate_plot(X_n, X_a, L_a, 10)

# GS_plot(X_a1, X_a2, X_a3, X_a4, X_a5, X_a6, X_a7, X_a8, gs1, gs2, gs3, gs4, gs5, gs6, gs7, gs8)
GS_plotI(X_a1, X_a2, X_a3, X_a4, gs1, gs2, gs3, gs4)
GS_plotII(X_a5, X_a6, X_a7, X_a8, gs5, gs6, gs7, gs8)

# MMD_plot(mmd_0, mmd_1)