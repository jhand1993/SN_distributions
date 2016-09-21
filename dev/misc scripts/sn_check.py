import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

testname = 'WFIRST'
columns = tuple(list(range(1, 5)) + list(range(6, 25)))
snfit = np.loadtxt('jsh_WFIRST_snfit.fitres', dtype=float, skiprows=14, usecols=columns)
host_logmass = np.copy(snfit[:, 9])
zCMB = np.copy(snfit[:, 5])
x1 = np.copy(snfit[:, 16])
c = np.copy(snfit[:, 18])

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
print('covariance:')
print('zCMB: ' + str(zCMBcov[0, 1]), 'x1: ' + str(x1cov[0, 1]), 'c: ' + str(ccov[0, 1]))
print('host_logmass variance: ' + str(zCMBcov[0, 0]))
print('zCMB variance:', zCMBcov[1, 1], '  x1 variance', x1cov[1, 1], '  c variance:', ccov[1, 1])
gammahat = stats.skew(host_logmass)
print('host_logmass skew: ' + str(gammahat))


def avgbins(arrx, arry, step, sem=False):
    start = int(min(arrx))
    stop = int(max(arrx)) + 1
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


zbins, bins = avgbins(host_logmass, zCMB, .5)
x1bins, bins = avgbins(host_logmass, x1, .5)
cbins, bins = avgbins(host_logmass, c, .5)

os.chdir('/Users/jaredhand/WFIRST_research/SN_distributions/plots')
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 16.0
fig_size[1] = 8.0

xlim = [7., 13.5]
plt.xlabel('host_logmass')
plt.ylabel('zCMB')
plt.xlim(xlim[0], xlim[1])
plt.scatter(host_logmass, zCMB, c='blue', label='sim', alpha=0.5)
plt.scatter(bins, zbins, c='red', marker='D', s=50,  label='bins')
plt.legend()
plt.savefig('%sCMB_scatter.png' % testname)
plt.show()

plt.xlabel('host_logmass')
plt.ylabel('x1')
plt.xlim(xlim[0], xlim[1])
plt.scatter(host_logmass, x1, c='blue', label='sim', alpha=0.5)
plt.scatter(bins, x1bins, c='red', marker='D', s=50, label='bins')
plt.legend()
plt.savefig('%sx1_scatter.png' % testname)
plt.show()

plt.xlabel('host_logmass')
plt.ylabel('c')
plt.xlim(xlim[0], xlim[1])
plt.scatter(host_logmass, c, c='blue', label='sim', alpha=0.5)
plt.scatter(bins, cbins, c='red', marker='D', s=50, label='bins')
plt.legend()
plt.savefig('%sc_scatter.png' % testname)
plt.show()

plt.xlim(xlim[0], xlim[1])
plt.hist(host_logmass, 20, normed=True, alpha=.8)
plt.ylabel('Count Density')
plt.xlabel('HOST_LOGMASS')
plt.savefig('%shost_logmass_hist.png' % testname)
plt.show()

