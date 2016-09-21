import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os

# import dragan data
os.chdir('/Users/jaredhand/WFIRST_research/SN_distributions/dragan/')
# columns = tuple(list(range(2, 5)) + list(range(6, 40)))
# dragan = np.loadtxt('FITOPT000+SALT2mu.FITRES', dtype=float, skiprows=18, usecols=columns)
columns = tuple(list(range(2, 5)) + list(range(6, 25)))
dragan = np.loadtxt('dragan2.FITRES', dtype=float, skiprows=14, usecols=columns)
datahostmass = np.copy(dragan[:, 8])
# datahostmasserr = np.copy(dragan[:, 9])
datazcmb = np.copy(dragan[:, 4])
# datazcmberr = np.copy(dragan[:, 5])
datax1 = np.copy(dragan[:, 15])
# datax1err = np.copy(dragan[:, 16])
datac = np.copy(dragan[:, 17])
# datacerr = np.copy(dragan[:, 18])
# datamures = np.copy(dragan[:, 35])
# datamureserr = np.copy(dragan[:, 33])
datacidint = np.copy(dragan[:, 1])


# only use CIDint = 1 from dragan data for now...
bad_cidint = np.asarray([x == 1 for x in datacidint])
datahostmass = datahostmass[bad_cidint]
# datahostmasserr = datahostmasserr[bad_cidint]
datazcmb = datazcmb[bad_cidint]
# datazcmberr = datazcmberr[bad_cidint]
datax1 = datax1[bad_cidint]
# datax1err = datax1err[bad_cidint]
datac = datac[bad_cidint]
# datacerr = datacerr[bad_cidint]
# datamures = datamures[bad_cidint]
# datamureserr = datamureserr[bad_cidint]

# remove host_logmass < 7 from dragan data.
bad_logmass = np.asarray([x >= 7. for x in datahostmass])
datahostmass = datahostmass[bad_logmass]
# datahostmasserr = datahostmasserr[bad_logmass]
datazcmb = datazcmb[bad_logmass]
# datazcmberr = datazcmberr[bad_logmass]
datax1 = datax1[bad_logmass]
# datax1err = datax1err[bad_logmass]
datac = datac[bad_logmass]
# datacerr = datacerr[bad_logmass]
# datamures = datamures[bad_logmass]
# datamureserr = datamureserr[bad_logmass]

"""
for i in range(len(datamures)):
    if abs(datamures[i]) > 5:
        print(i + 15,
              'mures:', datamures[i], 'hostmass:', datahostmass[i], 'z:', datazcmb[i], 'c:', datac[i], 'x1:', datax1[i])
"""
"""
# remove mures > abs(10) from dragan data
bad_mures = np.asarray([x < 10 for x in np.abs(datamures)])
datahostmass = datahostmass[bad_mures]
# datahostmasserr = datahostmasserr[bad_mures]
datazcmb = datazcmb[bad_mures]
# datazcmberr = datazcmberr[bad_mures]
datax1 = datax1[bad_mures]
# datax1err = datax1err[bad_mures]
datac = datac[bad_mures]
# datacerr = datacerr[bad_mures]
# datamures = datamures[bad_mures]
"""
# import sim data
os.chdir('/Users/jaredhand/WFIRST_research/SN_distributions/')
columns = tuple(list(range(1, 5)) + list(range(6, 65)))
data = np.loadtxt('jsh_WFIRST_snfit.fitres', dtype=float, skiprows=12, usecols=columns)
# data_var = np.genfromtxt('snfit_distTEST06_2000.fitres', dtype=str, skip_header=1, skip_footer=2000, usecols=columns)
mures = np.copy(data[:, 59])
mureserr = np.copy(data[:, 57])
hostmass = np.copy(data[:, 9])
hostmasserr = np.copy(data[:, 10])
zcmb = np.copy(data[:, 5])
zcmberr = np.copy(data[:, 6])
x1 = np.copy(data[:, 16])
x1err = np.copy(data[:, 17])
c = np.copy(data[:, 18])
cerr = np.copy(data[:, 19])
# sim_mu = np.copy(data[:, 50])


def avgbins(arrx, arry, step, sem=False):
    """
    returns average y values binned with respect to x for specified step size.
    :param arrx:
    :param arry:
    :param step:
    :param sem:
    :return:
    """
    start = min(arrx)
    stop = max(arrx)
    bins = np.arange(start, stop, step)
    meanbins = []
    errbins = []
    for i in bins:
        active = np.asarray([(i + step) > x >= i for x in arrx])
        y = np.copy(arry[active])
        if len(y) == 0:
            meanbins.append(0)
        else:
            meanbins.append(np.mean(y))
        if sem:
            errbins.append(stats.sem(y))
    bins_centered = np.asarray([x + step / 2 for x in bins])
    if sem:
        return np.asarray(meanbins), bins_centered, np.asarray(errbins)
    else:
        return np.asarray(meanbins), bins_centered

zlen = 6
zbins = np.linspace(min(zcmb), max(zcmb) - 0.05, zlen)
zstep = zbins[1] - zbins[0]

zbin0 = np.asarray([zbins[0] <= x < zbins[1] for x in zcmb])
zbin1 = np.asarray([zbins[1] <= x < zbins[2] for x in zcmb])
zbin2 = np.asarray([zbins[2] <= x < zbins[3] for x in zcmb])
zbin3 = np.asarray([zbins[3] <= x < zbins[4] for x in zcmb])
zbin4 = np.asarray([zbins[4] <= x < zbins[5] for x in zcmb])
zbin5 = np.asarray([zbins[5] <= x < zbins[5] + zstep for x in zcmb])

datazbin0 = np.asarray([zbins[0] <= x < zbins[1] for x in datazcmb])
datazbin1 = np.asarray([zbins[1] <= x < zbins[2] for x in datazcmb])
datazbin2 = np.asarray([zbins[2] <= x < zbins[3] for x in datazcmb])
datazbin3 = np.asarray([zbins[3] <= x < zbins[4] for x in datazcmb])
datazbin4 = np.asarray([zbins[4] <= x < zbins[5] for x in datazcmb])
datazbin5 = np.asarray([zbins[5] <= x < zbins[5] + zstep for x in datazcmb])

simzbins = [zbin0, zbin1, zbin2, zbin3, zbin4, zbin5]
datazbins = [datazbin0, datazbin1, datazbin2, datazbin3, datazbin4, datazbin5]

for z in range(zlen):
    covmat = np.cov(hostmass[simzbins[z]], x1[simzbins[z]])
    datacovmat = np.cov(datahostmass[datazbins[z]], datax1[datazbins[z]])
    print('Redshift from ' + str(zbins[z]) + ' to ' + str(zbins[z] + zstep))
    print('x1 mass cov:')
    print('sim:', covmat[0, 1], ' obs:', datacovmat[0, 1])
    print('x1 mass corr:')
    print('sim: ', covmat[0, 1] / np.sqrt(covmat[0, 0] * covmat[1, 1]), ' obs:',
          datacovmat[0, 1] / np.sqrt(datacovmat[0, 0] * datacovmat[1, 1]))
    print('')

fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 16.0
fig_size[1] = 8.0

plt.subplot(231)
plt.scatter(hostmass[zbin0], x1[zbin0], alpha=0.6, color='blue')
plt.scatter(datahostmass[datazbin0], datax1[datazbin0], alpha=0.8, color='red', marker='v')
plt.title('z = ' + str(zbins[0]))
plt.ylabel('x1')
plt.xlim(7, 13)
plt.ylim(-4, 3)
plt.subplot(232)
plt.scatter(hostmass[zbin1], x1[zbin1], alpha=0.6, color='blue', label='sim')
plt.scatter(datahostmass[datazbin1], datax1[datazbin1], alpha=0.8, color='red', marker='v', label='obs')
plt.title('z = ' + str(zbins[1]))
# plt.legend()
plt.xlim(7, 13)
plt.ylim(-4, 3)
plt.subplot(233)
plt.scatter(hostmass[zbin2], x1[zbin2], alpha=0.6, color='blue', label='sim')
plt.scatter(datahostmass[datazbin2], datax1[datazbin2], alpha=0.8, color='red', marker='v', label='obs')
plt.title('z = ' + str(zbins[2]))
plt.legend()
plt.xlim(7, 13)
plt.ylim(-4, 3)
plt.subplot(234)
plt.scatter(hostmass[zbin3], x1[zbin3], alpha=0.6, color='blue')
plt.scatter(datahostmass[datazbin3], datax1[datazbin3], alpha=0.8, color='red', marker='v')
plt.title('z = ' + str(zbins[3]))
plt.xlim(7, 13)
plt.ylim(-4, 3)
plt.subplot(235)
plt.scatter(hostmass[zbin4], x1[zbin4], alpha=0.6, color='blue')
plt.scatter(datahostmass[datazbin4], datax1[datazbin4], alpha=0.8, color='red', marker='v')
plt.title('z = ' + str(zbins[4]))
plt.xlabel('host mass')
plt.xlim(7, 13)
plt.ylim(-4, 3)
plt.subplot(236)
plt.scatter(hostmass[zbin5], x1[zbin5], alpha=0.6, color='blue')
plt.scatter(datahostmass[datazbin5], datax1[datazbin5], alpha=0.8, color='red', marker='v')
plt.title('z = ' + str(zbins[5]))
plt.xlim(7, 13)
plt.ylim(-4, 3)
plt.show()

