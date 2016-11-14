import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

os.chdir('/Users/jaredhand/WFIRST_research/SN_distributions/host_Sims/versions/GRID/hostlib/SDSS/')
columns = tuple(list(range(1, 8)))
data = np.loadtxt('JSH_WFIRST_SDSS_G10.HOSTLIB', skiprows=29, usecols=columns)
os.chdir('/Users/jaredhand/WFIRST_research/SN_distributions/')
host_logmass = np.copy(data[:, 4])
x1 = np.copy(data[:, 3])
c = np.copy(data[:, 2])
z = np.copy(data[:, 1])

# Dragan data
os.chdir('/Users/jaredhand/WFIRST_research/SN_distributions/host_sims/versions/GRID/dragan/')
columns = tuple(list(range(2, 5)) + list(range(6, 25)))
data_snfit = np.loadtxt('dragan2.FITRES', dtype=float, skiprows=14, usecols=columns)
data_host_logmass = np.copy(data_snfit[:, 8])
data_zCMB = np.copy(data_snfit[:, 4])
data_x1 = np.copy(data_snfit[:, 15])
data_c = np.copy(data_snfit[:, 17])
cidint = np.copy(data_snfit[:, 1])
os.chdir('/Users/jaredhand/WFIRST_research/SN_distributions/dev/hostlib/snls')

# only use CIDint = 1
bad_cidint = np.asarray([x == 1 for x in cidint])
data_host_logmass = data_host_logmass[bad_cidint]
data_zCMB = data_zCMB[bad_cidint]
data_x1 = data_x1[bad_cidint]
data_c = data_c[bad_cidint]

# remove host_logmass < 7 from dragan data.
bad_logmass = np.asarray([x >= 7. for x in data_host_logmass])
data_host_logmass = data_host_logmass[bad_logmass]
data_zCMB = data_zCMB[bad_logmass]
data_x1 = data_x1[bad_logmass]
data_c = data_c[bad_logmass]

print(np.mean(data_host_logmass))

def digitize(arr1, arr2, bounds=False, step1=1., step2=1., norm=False):
    """
    Finds weights of arr1 vs arr2 2D plot
    :param arr1: x-axis arr
    :param arr2: y-axis arr
    :param bounds: If False, max and min values of arr1, arr2 used for bounds.  Otherwise user can specify bounds as
        list of length 4.
    :param step1: x-axis resolution.  Default is 1.
    :param step2: y-axis resolution.  Default is 1.
    :param norm: If true, returned array is normalized.  Default is False
    :return:
    """
    if not bounds:
        max1 = int(max(arr1)) + 1
        min1 = int(min(arr1)) - 1
        max2 = int(max(arr2)) + 1
        min2 = int(min(arr2)) - 1
    elif len(bounds) == 4:
        min1, max1 = bounds[0], bounds[1]
        min2, max2 = bounds[2], bounds[3]
    bins1 = np.arange(min1, max1, step1)
    bins2 = np.arange(min2, max2, step2)
    cells = np.zeros((len(bins2), len(bins1)), dtype=float)
    superarr = np.vstack((arr1, arr2)).T
    for i in range(len(bins2)):
        for j in range(len(bins1)):
            for k in superarr:
                if bins2[i] <= k[1] < bins2[i] + step2 and bins1[j] <= k[0] < bins1[j] + step1:
                    cells[i, j] += 1.
    if norm:
        return cells / np.sum(cells)
    else:
        return cells

x1massmap = digitize(host_logmass, x1, bounds=[7., 12., -4., 3.], step1=0.5, step2=0.5)
cmassmap = digitize(host_logmass, c, bounds=[7., 12., -0.3, 0.3], step1=0.5, step2=0.05)
datax1massmap = digitize(data_host_logmass, data_x1, bounds=[7., 12., -4., 3.], step1=0.5, step2=0.5)
datacmassmap = digitize(data_host_logmass, data_c, bounds=[7., 12., -0.3, 0.3], step1=0.5, step2=0.5)
# print(datax1massmap)
# print(x1massmap)


"""
columns = tuple(list(range(1, 4)) + list(range(5, 25)))
snfit = np.loadtxt('snfit06.fitres', dtype=float, skiprows=14, usecols=columns)
simhost_logmass = np.copy(snfit[:, 8])
simzCMB = np.copy(snfit[:, 4])
simx1 = np.copy(snfit[:, 15])
simc = np.copy(snfit[:, 17])
"""

p = np.cov(x1, host_logmass)
print('x1 correlation:', p[1, 0] / np.sqrt(p[1, 1] * p[0, 0]))
print('x1 cov:', p[1, 0])
gammahat = stats.skew(host_logmass)
print('host_logmass skew: ' + str(gammahat))
abvten = np.asarray([x >= 10 for x in host_logmass])
belten = np.asarray([x < 10 for x in host_logmass])
cabv = c[abvten]
cbel = c[belten]
# dataabvten = np.asarray([x >= 10 for x in data_host_logmass])
# databelten = np.asarray([x < 10 for x in data_host_logmass])
# datacabv = c[dataabvten]
# datacbel = c[databelten]
# print('mean mass:', np.mean(host_logmass))
# print('hostlib:', np.mean(cbel), np.mean(cabv))
# print('data:', np.mean(datacbel), np.mean(datacabv))

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 16.0
fig_size[1] = 8.0

plt.scatter(host_logmass, x1, alpha=0.6, label='hostlib')
plt.scatter(data_host_logmass, data_x1, alpha=0.8, marker='v', color='red')
plt.xlabel('host mass')
plt.ylabel('x1')
plt.legend()
plt.show()

plt.scatter(host_logmass, c, alpha=0.6, label='hostlib')
plt.scatter(data_host_logmass, data_c, alpha=0.8, marker='v', color='red')
plt.xlabel('host mass')
plt.ylabel('c')
plt.legend()
plt.show()

# plt.hist(simhost_logmass, 100, normed=True, alpha=0.4, label='after')
plt.hist(host_logmass, 20, normed=True, alpha=0.4)
plt.hist(data_host_logmass, 20, normed=True, alpha=0.5, color='red')
plt.xlabel('host mass')
plt.xlim(7, 13.5)
plt.show()

plt.hist(c, 20, normed=True, alpha=0.4, label='sim')
# plt.hist(data_c, 20, color='red', normed=True, alpha=0.4, label='obs')
plt.xlabel('c')
plt.legend()
plt.show()

plt.hist(x1, 20, normed=True, alpha=0.4, label='sim')
# plt.hist(data_x1, 20, color='red', normed=True, alpha=0.4, label='obs')
plt.xlabel('x1')
plt.legend()
plt.show()


# plt.hist(x1, 100, normed=True, alpha=0.4)
# plt.xlabel('x1')
# plt.show()

# plt.hist(simx1, 100, normed=True, alpha=0.3, label='after')
# plt.hist(x1, 100, normed=True, alpha=0.3, label='before')
# plt.xlabel('x1')
# plt.legend()
# plt.show()

# heat maps
x1bounds = [7., 12., -4., 3.]
cbounds = [7., 12., -0.3, 0.3]
massrange = np.arange(x1bounds[0], x1bounds[1], 0.5)
x1range = np.arange(x1bounds[2], x1bounds[3], 0.5)
crange = np.arange(cbounds[2], cbounds[3], 0.05)

plt.pcolormesh(x1massmap)
plt.xticks(np.arange(len(massrange)), massrange)
plt.yticks(np.arange(len(x1range)), x1range)
plt.title('hostlib')
plt.ylabel('x1')
plt.xlabel('host mass')
plt.colorbar()
plt.show()

plt.pcolormesh(cmassmap)
plt.xticks(np.arange(len(massrange)), massrange)
plt.yticks(np.arange(len(crange)), crange)
plt.title('hostlib')
plt.ylabel('c')
plt.xlabel('host mass')
plt.colorbar()
plt.show()
