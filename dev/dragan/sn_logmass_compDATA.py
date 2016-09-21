import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

columns = tuple(list(range(2, 5)) + list(range(6, 25)))
snfit = np.loadtxt('dragan2.fitres', dtype=float, skiprows=14, usecols=columns)
host_logmass = np.copy(snfit[:, 8])
zCMB = np.copy(snfit[:, 4])
x1 = np.copy(snfit[:, 15])
c = np.copy(snfit[:, 17])
cidint = np.copy(snfit[:, 1])

# only use CIDint = 1
bad_cidint = np.asarray([x == 1 for x in cidint])
host_logmass = host_logmass[bad_cidint]
zCMB = zCMB[bad_cidint]
x1 = x1[bad_cidint]
c = c[bad_cidint]

# remove host_logmass < 7 from data.
bad_logmass = np.asarray([x >= 7. for x in host_logmass])
host_logmass = host_logmass[bad_logmass]
zCMB = zCMB[bad_logmass]
x1 = x1[bad_logmass]
c = c[bad_logmass]

ndist = lambda x: (1 / np.sqrt(2 * np.pi * 1.5**2)) * np.exp(-(x - 10.)**2 / (2 * 1.5**2))

x_arr = np.linspace(4., 16., 1000)
p_arr = ndist(x_arr)

"""
for i in range(len(host_logmass)):
    if host_logmass[i] == -9.:
        print(snfit[i, 1])


bytes_snfit = np.loadtxt('dragan2.fitres', skiprows=14, dtype=bytes)
for i in range(len(bytes_snfit[:, 0])):
    if float((bytes_snfit[i, 11].decode('utf8'))) == -9.:
        print(bytes_snfit[i, 1].decode('utf8') + '   ', bytes_snfit[i, 3].decode('utf8'))
"""

zCMBcov = np.cov(host_logmass, zCMB)
x1cov = np.cov(host_logmass, x1)
ccov = np.cov(host_logmass, c)
print('covariances:')
print('zCMB: ' + str(zCMBcov[0, 1]), 'x1: ' + str(x1cov[0, 1]), 'c: ' + str(ccov[0, 1]))
print('host_logmass variance: ' + str(zCMBcov[0, 0]))
print('zCMB variance:', zCMBcov[1, 1], '  x1 variance', x1cov[1, 1], '  c variance:', ccov[1, 1])
gammahat = stats.skew(host_logmass)
print('host_logmass skew: ' + str(gammahat))
ccorr = ccov[1, 0] / np.sqrt(ccov[1, 1] * ccov[0, 0])
print('c correlation:', ccorr)
x1corr = x1cov[1, 0] / np.sqrt(x1cov[1, 1] * x1cov[0, 0])
print('x1 correlation:', x1corr)

# print out c avg for mass < 10 and mass > 10, respectively
indexerabv = np.asarray([x >= 10 for x in host_logmass])
indexerbel = np.asarray([x < 10 for x in host_logmass])
cabv = c[indexerabv]
cbel = c[indexerbel]
print('c mean:', np.mean(c), 'cbel mean:', np.mean(cbel), 'cabv mean:', np.mean(cabv))


def avgbins(arrx, arry, start, stop,  step):
    arrlen = len(arrx)
    bins = np.arange(start, stop, step)
    arr_avgbins = []
    for i in bins:
        count = []
        for j in range(arrlen):
            if i <= arrx[j] < (i + step):
                count.append(arry[j])
        if len(count) == 0:
            pass
        else:
            avg = sum(count) / len(count)
            arr_avgbins.append(avg)
    return np.asarray(arr_avgbins)


def avgbins2(arrx, arry, step, sem=False):
    # returns average y values binned with respect to x and specified step size.
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

bins = np.arange(8.5, 12, 0.5)
zbins = avgbins(host_logmass, zCMB, 8, 12, 0.5)
x1bins = avgbins(host_logmass, x1, 8, 12, 0.5)
cbins = avgbins(host_logmass, c, 8, 12, 0.5)
host_mass = np.array([10**x for x in host_logmass])
cbins2, x1bins2 = avgbins2(x1, c, 0.5)
print(len(bins), len(zbins))

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 16.0
fig_size[1] = 8.0

os.chdir('/Users/jaredhand/WFIRST_research/SN_distributions/dragan/dragan_plots/cid1_v5')
plt.xlabel('host_logmass')
plt.ylabel('zCMB')
plt.xlim(7., 13.5)
plt.scatter(host_logmass, zCMB, c='blue', label='data', alpha=0.5)
plt.scatter(bins, zbins, c='red', marker='D', s=50,  label='bins', alpha=0.9)
plt.legend()
plt.savefig('Dragan_zCMB_scatter.png')
plt.show()

plt.xlabel('host_logmass')
plt.ylabel('x1')
plt.xlim(7., 13.5)
plt.scatter(host_logmass, x1, c='blue', label='data', alpha=0.5)
plt.scatter(bins, x1bins, c='red', marker='D', s=50, label='bins', alpha=0.9)
plt.legend()
plt.savefig('Dragan_x1_scatter.png')
plt.show()

plt.xlabel('host_logmass')
plt.ylabel('c')
plt.xlim(7., 13.5)
plt.scatter(host_logmass, c, c='blue', label='data', alpha=0.5)
plt.scatter(bins, cbins, c='red', marker='D', s=50, label='bins', alpha=0.9)
plt.legend()
plt.savefig('Dragan_c_scatter.png')
plt.show()

plt.xlim(7., 13.5)
plt.hist(host_logmass, 20, normed=True, alpha=.8)
# plt.plot(x_arr, p_arr)
plt.ylabel('Count Density')
plt.xlabel('HOST_LOGMASS')
plt.savefig('Dragan_host_logmass_hist.png')
plt.show()

plt.scatter(x1, c, alpha=0.5, label='data')
plt.scatter(x1bins2, cbins2, alpha=1, color='red', marker='D', s=50, label='bins')
plt.ylabel('c')
plt.xlabel('x1')
plt.legend()
plt.savefig('x1_c.png')
plt.show()

