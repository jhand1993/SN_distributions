import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os

# import dragan data
# datafitres = 'dragan2.fitres'
datafitres = 'supercal_vH0.fitres'
os.chdir('/Users/jaredhand/WFIRST_research/SN_distributions/dev/dragan')
with open(datafitres, 'r') as f:
    datavars = f.readlines()[1]
with open(datafitres, 'r') as f:
    datavarscount = int(f.readlines()[0].split(' ')[1]) - 2
datavars = datavars.split()
datavars.remove('VARNAMES:')
datavars.remove('FIELD')
datavars.remove('CID')
print(datavarscount, len(datavars))
columns = tuple(list(range(2, 5)) + list(range(6, 42)))
dragan = np.loadtxt(datafitres, dtype=float, skiprows=18, usecols=columns)
for i in range(datavarscount):
    if datavars[i] == 'IDSURVEY':
        dataidcol = i
        datacidint = np.copy(dragan[:, i])
    elif datavars[i] == 'zCMB':
        datazcol = i
        datazcmb = np.copy(dragan[:, i])
    elif datavars[i] == 'z':
        datazcol = i
        datazcmb = np.copy(dragan[:, i])
    elif datavars[i] == 'HOST_LOGMASS':
        datahostmass = np.copy(dragan[:, i])
    elif datavars[i] == 'c':
        datac = np.copy(dragan[:, i])
    elif datavars[i] == 'x1':
        datax1 = np.copy(dragan[:, i])
    elif datavars[i] == 'MURES':
        datamures = np.copy(dragan[:, i])
    elif datavars[i] == 'HOST_LOGMASS_ERR':
        datahostmasserr = np.copy(dragan[:, i])


# only use CIDint = 1,4 from dragan data for now...
bad_cidint = np.asarray([x == 1 or x==4 for x in datacidint])
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
datahostmasserr = datahostmasserr[bad_logmass]
datazcmb = datazcmb[bad_logmass]
# datazcmberr = datazcmberr[bad_logmass]
datax1 = datax1[bad_logmass]
# datax1err = datax1err[bad_logmass]
datac = datac[bad_logmass]
# datacerr = datacerr[bad_logmass]
datamures = datamures[bad_logmass]
# datamureserr = datamureserr[bad_logmass]

# remove mures > abs(10) from dragan data
bad_mures = np.asarray([x < 10 for x in np.abs(datamures)])
datahostmass = datahostmass[bad_mures]
datahostmasserr = datahostmasserr[bad_mures]
datazcmb = datazcmb[bad_mures]
# datazcmberr = datazcmberr[bad_mures]
datax1 = datax1[bad_mures]
# datax1err = datax1err[bad_mures]
datac = datac[bad_mures]
# datacerr = datacerr[bad_mures]
# datamures = datamures[bad_mures]

# remove host mass error > 15 > 0.0001 from dragan data
bad_masserr = np.asarray([0.0001 < x < 15 for x in datahostmasserr])
datahostmass = datahostmass[bad_masserr]
datahostmasserr = datahostmasserr[bad_masserr]
datazcmb = datazcmb[bad_masserr]
# datazcmberr = datazcmberr[bad_masserr]
datax1 = datax1[bad_masserr]
# datax1err = datax1err[bad_masserr]
datac = datac[bad_masserr]
# datacerr = datacerr[bad_masserr]
# datamures = datamures[bad_masserr]


# import sim data
fitres = 'composite.fitres'
os.chdir('/Users/jaredhand/WFIRST_research/SN_distributions/dev/fitres/composite')
with open(fitres, 'r') as f:
    vars = f.readlines()[1]
with open(fitres, 'r') as f:
    varscount = int(f.readlines()[0].split(' ')[1]) - 1
vars = vars.split()
vars.remove('VARNAMES:')
# vars.remove('FIELD')
columns = tuple(list(range(1, 55)))
data = np.loadtxt(fitres, dtype=float, skiprows=2, usecols=columns)
for i in range(varscount):
    if vars[i] == 'MURES':
        mures = np.copy(data[:, i])
    elif vars[i] == 'HOST_LOGMASS':
        hostmass = np.copy(data[:, i])
    elif vars[i] == 'zCMB':
        zcmb = np.copy(data[:, i])
    elif vars[i] == 'x1':
        x1 = np.copy(data[:, i])
    elif vars[i] == 'c':
        c = np.copy(data[:, i])
    elif vars[i] == 'HOST_LOGMASS_ERR':
        hostmasserr = np.copy(data[:, i])


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


def medbins(arrx, arry, step, sem=False):
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
    medbins = []
    errbins = []
    for i in bins:
        active = np.asarray([(i + step) > x >= i for x in arrx])
        y = np.copy(arry[active])
        if len(y) == 0:
            medbins.append(0)
        else:
            medbins.append(np.nanmedian(y))
        if sem:
            errbins.append(stats.sem(y))
    bins_centered = np.asarray([x + step / 2 for x in bins])
    if sem:
        return np.asarray(medbins), bins_centered, np.asarray(errbins)
    else:
        return np.asarray(medbins), bins_centered

print(np.mean(hostmass), np.mean(datahostmass))
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
    print('Redshift from ' + str(zbins[int(z)]) + ' to ' + str(zbins[int(z)] + zstep))
    print('mass mean:')
    print('sim:', np.nanmean(hostmass[zbins[int(z)]]), ' obs:', np.nanmean(datahostmass[datazbins[int(z)]]))
    print('mass skew:')
    print('sim: ', stats.skew(hostmass[zbins[z]]), ' obs:', stats.skew(datahostmass[datazbins[z]]))
    print('')

print(len(zcmb), len(hostmasserr))
hostmasserrbins, masszbins = medbins(zcmb, hostmasserr, 0.05)
datahostmasserrbins, datamasszbins = medbins(datazcmb, datahostmasserr, 0.05)

fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 16.0
fig_size[1] = 8.0

plt.subplot(231)
# plt.hist(hostmass[zbin0], alpha=0.6, color='blue', normed=True)
plt.hist(datahostmass[datazbin0], alpha=0.5, color='red', normed=True)
plt.title('z = ' + str(zbins[0]))
plt.xlim(7, 13)
plt.subplot(232)
# plt.hist(hostmass[zbin1], alpha=0.6, color='blue', label='sim', normed=True)
plt.hist(datahostmass[datazbin1], alpha=0.5, color='red', label='obs', normed=True)
plt.title('z = ' + str(zbins[1]))
plt.xlim(7, 13)
plt.subplot(233)
# plt.hist(hostmass[zbin2], alpha=0.6, color='blue', label='sim', normed=True)
plt.hist(datahostmass[datazbin2], alpha=0.5, color='red', label='obs', normed=True)
plt.title('z = ' + str(zbins[2]))
plt.legend()
plt.xlim(7, 13)
plt.subplot(234)
# plt.hist(hostmass[zbin3], alpha=0.6, color='blue', normed=True)
plt.hist(datahostmass[datazbin3], alpha=0.5, color='red', normed=True)
plt.title('z = ' + str(zbins[3]))
plt.xlim(7, 13)
plt.subplot(235)
# plt.hist(hostmass[zbin4], alpha=0.6, color='blue', normed=True)
plt.hist(datahostmass[datazbin4], alpha=0.5, color='red', normed=True)
plt.title('z = ' + str(zbins[4]))
plt.xlabel('host mass')
plt.xlim(7, 13)
plt.subplot(236)
# plt.hist(hostmass[zbin5], alpha=0.6, color='blue', normed=True)
plt.hist(datahostmass[datazbin5], alpha=0.5, color='red', normed=True)
plt.title('z = ' + str(zbins[5]))
plt.xlim(7, 13)
plt.show()

# plt.plot(masszbins, hostmasserrbins, color='blue', label='Sim')
plt.plot(datamasszbins, datahostmasserrbins, color='red', label='Obs')
# plt.scatter(masszbins, hostmasserrbins, color='blue')
plt.scatter(datazcmb, datahostmasserr, color='red')
plt.xlim(0., 1)
plt.xlabel('Redshift')
plt.ylabel('Avg host mass error')
plt.legend()
plt.show()

