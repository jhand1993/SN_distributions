import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from config import *

surv = 'SDSS'

# import dragan data
datafitres = 'dragan2.fitres'
os.chdir(homedir + '/dragan')
with open(datafitres, 'r') as f:
    datavars = f.readlines()[1]
with open(datafitres, 'r') as f:
    datavarscount = int(f.readlines()[0].split(' ')[1]) - 2
datavars = datavars.split()
datavars.remove('VARNAMES:')
datavars.remove('FIELD')
datavars.remove('CID')
columns = tuple(list(range(2, 5)) + list(range(6, 42)))
dragan = np.loadtxt(datafitres, dtype=float, skiprows=18, usecols=columns)
for i in range(datavarscount):
    if datavars[i] == 'IDSURVEY':
        dataidcol = i
        datacidint = np.copy(dragan[:, i])
    elif datavars[i] == 'zCMB':
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

newdragan = np.copy(dragan)

# only use z<0.1 unless cid=1,4,15
bad_z = np.asarray([x <= 0.1 for x in datazcmb])
bad_cidint = np.asarray([(x == 1 or x == 4) for x in datacidint])
bad_zandcid = np.logical_or(bad_z, bad_cidint)
datahostmass = datahostmass[bad_zandcid]
datazcmb = datazcmb[bad_zandcid]
datax1 = datax1[bad_zandcid]
datac = datac[bad_zandcid]
datamures = datamures[bad_zandcid]
newdragan = newdragan[bad_zandcid]
datacidint = datacidint[bad_zandcid]

# remove host_logmass < 7 from dragan data.
bad_logmass = np.asarray([x >= 7. for x in datahostmass])
datahostmass = datahostmass[bad_logmass]
datazcmb = datazcmb[bad_logmass]
datax1 = datax1[bad_logmass]
datac = datac[bad_logmass]
datamures = datamures[bad_logmass]
datacidint = datacidint[bad_logmass]
newdragan = newdragan[bad_logmass]

# remove mures > abs(10) from dragan data
bad_mures = np.asarray([x < 10 for x in np.abs(datamures)])
datahostmass = datahostmass[bad_mures]
datazcmb = datazcmb[bad_mures]
datax1 = datax1[bad_mures]
datac = datac[bad_mures]
datamures = datamures[bad_mures]
datacidint = datacidint[bad_mures]
newdragan = newdragan[bad_mures]

# import sim data
fitres = '{}_WFIRST_{}.fitres'.format(init, surv)
os.chdir(homedir)
with open(fitres, 'r') as f:
    vars = f.readlines()[1]
with open(fitres, 'r') as f:
    varscount = int(f.readlines()[0].split(' ')[1]) - 2
vars = vars.split()
vars.remove('VARNAMES:')
vars.remove('FIELD')
columns = tuple(list(range(1, 5)) + list(range(6, 56)))
data = np.loadtxt(fitres, dtype=float, skiprows=12, usecols=columns)
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

# find mures above and below hostmass=10:
aboveten = np.asarray([x >= 10 for x in hostmass])
belowten = np.asarray([x < 10 for x in hostmass])

# find avg and std of sim results
aboveavg = np.median(mures[aboveten])
belowavg = np.median(mures[belowten])
muresavg = np.median(mures)
print('SIM')
print('Mass greater than 10:', aboveavg)
print('Mass less than 10:', belowavg)
print('median:', muresavg)
print('std:', np.std(mures), 'std above:', np.std(mures[aboveten]), 'std below:', np.std(mures[belowten]))
# print(np.median(np.abs(mures)))
# print(max(mures), min(mures))

# find mures above and below hostmass=10 for dragan data
dataaboveten = np.asarray([x >= 10 for x in datahostmass])
databelowten = np.asarray([x < 10 for x in datahostmass])

# find avg and std of dragan data results
dataaboveavg = np.median(datamures[dataaboveten])
databelowavg = np.median(datamures[databelowten])
datamuresavg = np.median(datamures)
print('DATA')
print('Mass greater than 10:', dataaboveavg)
print('Mass less than 10:', databelowavg)
print('median:', datamuresavg)
print('std:', np.std(datamures), 'std above:', np.std(datamures[dataaboveten]),
      'std below:', np.std(datamures[databelowten]))
# print(np.median(np.abs(datamures)))
# print(max(datamures), min(datamures))

# split up sim mures, x1, c and zcmb based on mass greater or less than 10
muresabv = np.asarray(mures[aboveten])
muresbel = np.asarray(mures[belowten])
cabv = np.asarray(c[aboveten])
cbel = np.asarray(c[belowten])
x1abv = np.asarray(x1[aboveten])
x1bel = np.asarray(x1[belowten])
zabv = np.asarray(zcmb[aboveten])
zbel = np.asarray(zcmb[belowten])

# split up data mures, x1, c and zcmb based on mass greater or less than 10
datamuresabv = np.asarray(datamures[dataaboveten])
datamuresbel = np.asarray(datamures[databelowten])
datacabv = np.asarray(datac[dataaboveten])
datacbel = np.asarray(datac[databelowten])
datax1abv = np.asarray(datax1[dataaboveten])
datax1bel = np.asarray(datax1[databelowten])
datazabv = np.asarray(datazcmb[dataaboveten])
datazbel = np.asarray(datazcmb[databelowten])


def mass_params(massarr, zeroed=False):
    # determines mode, left std and right std of asym host mass distribution
    xbins = np.linspace(4, 20, 25)
    step = xbins[1] - xbins[0]
    mbins = []
    # count up bins of mass data to determine mode
    for i in xbins:
        count = 0
        for j in massarr:
            if i <= j < (i + step):
                count += 1
        mbins.append(count)
    mmax = 0
    maxpos = 0
    for i in range(len(mbins)):
        if mbins[i] > mmax:
            mmax = mbins[i]
            maxpos = i
    mode = xbins[maxpos] + step / 2
    maskbel = np.asarray([x < mode for x in massarr])
    maskabv = np.asarray([x >= mode for x in massarr])
    massabv = massarr[maskabv]
    massbel = massarr[maskbel]
    massruntot = 0
    massruntot2 = 0
    for i in massabv:
        massruntot += i
        massruntot2 += i**2
    posstd = np.sqrt(massruntot2 / len(massabv) - (massruntot / len(massabv))**2)
    massruntot = 0
    massruntot2 = 0
    for i in massbel:
        massruntot += i
        massruntot2 += i ** 2
    negstd = np.sqrt(massruntot2 / len(massbel) - (massruntot / len(massbel))**2)
    if zeroed:
        return[0, posstd, negstd]
    else:
        return [mode, posstd, negstd]

# get asym mass params for data and sim results:
mp = mass_params(hostmass)
datamp = mass_params(datahostmass)


def avgbins(arrx, arry, step, sem=False):
    """
    This function bins arrx in intervals of size step and returns average arry bins for said intervals along with
        arrx bins.  Can also return errors for bins.  Starting and stopping point of bins determined by min and max
        of arrx, respectively.
    :param arrx: x-axis data
    :param arry: y-axis data.  Average value in bins are calculated for arrx intervals.
    :param step: Size of arrx intervals.
    :param sem: If sem=true, third array is return that includes error calculation for average arry bins
    :return: binned average arry, binned arrx, error bins
    """
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


def digitize2D(arr1, arr2, bounds, step1=1., step2=1., norm=False):
    """
    Finds weights of arr1 vs arr2 2D plot
    :param arr1: x-axis arr
    :param arr2: y-axis arr
    :param bounds: user can specify bounds as list of length 4.
    :param step1: x-axis resolution.  Default is 1.
    :param step2: y-axis resolution.  Default is 1.
    :param norm: If true, returned array is normalized.  Default is False
    :return:
    """
    if len(bounds) == 4:
        min1, max1 = bounds[0], bounds[1]
        min2, max2 = bounds[2], bounds[3]
    else:
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

# make cells for heat map
massx1bounds = [7., 13., -4., 3.]
masscbounds = [7., 13, -0.3, 0.3]
x1massmap = digitize2D(hostmass, x1, massx1bounds)
cmassmap = digitize2D(hostmass, c, masscbounds, step1=0.5, step2=0.05)
datax1massmap = digitize2D(datahostmass, datax1, massx1bounds)
datacmassmap = digitize2D(datahostmass, datac, masscbounds, step1=0.5, step2=0.05)

# bin mures with respect to mass
muresbins2, massbins2, massbinserr2 = avgbins(hostmass, mures, 1, sem=True)
# bin mures with respect to z
muresabvbins, zabvbins, muresabvbinserr = avgbins(zabv, muresabv, 0.05, sem=True)
muresbelbins, zbelbins, muresbelbinserr = avgbins(zbel, muresbel, 0.05, sem=True)
muresbins, zbins, muresbinserr = avgbins(zcmb, mures, 0.05, sem=True)
muresabvbins[muresabvbins == 0] = np.nan
muresbelbins[muresbelbins == 0] = np.nan
muresbins[muresbins == 0] = np.nan
# bin c with respect to z
cabvbins, zabvbins, cabvbinserr = avgbins(zabv, cabv, 0.05, sem=True)
cbelbins, zbelbins, cbelbinserr = avgbins(zbel, cbel, 0.05, sem=True)
cbins, zbins, cbinserr = avgbins(zcmb, c, 0.05, sem=True)
cabvbins[cabvbins == 0] = np.nan
cbelbins[cbelbins == 0] = np.nan
cbins[cbins == 0] = np.nan
# bin x1 with respect to z
x1abvbins, zabvbins, x1abvbinserr = avgbins(zabv, x1abv, 0.05, sem=True)
x1belbins, zbelbins, x1belbinserr = avgbins(zbel, x1bel, 0.05, sem=True)
x1bins, zbins, x1binserr = avgbins(zcmb, x1, 0.05, sem=True)
x1abvbins[x1abvbins == 0] = np.nan
x1belbins[x1belbins == 0] = np.nan
x1bins[x1bins == 0] = np.nan

# bin mures with respect to mass from dragan
datamuresbins2, datamassbins2, datamassbinserr2 = avgbins(datahostmass, datamures, 1, sem=True)
# bin mures with respect to z from dragan
datamuresabvbins, datazabvbins, datamuresabvbinserr = avgbins(datazabv, datamuresabv, 0.05, sem=True)
datamuresbelbins, datazbelbins, datamuresbelbinserr = avgbins(datazbel, datamuresbel, 0.05, sem=True)
datamuresbins, datazbins, datamuresbinserr = avgbins(datazcmb, datamures, 0.05, sem=True)
datamuresabvbins[datamuresabvbins == 0] = np.nan
datamuresbelbins[datamuresbelbins == 0] = np.nan
datamuresbins[datamuresbins == 0] = np.nan
# bin c with respect to z from dragan
datacabvbins, datazabvbins, datacabvbinserr = avgbins(datazabv, datacabv, 0.05, sem=True)
datacbelbins, datazbelbins, datacbelbinserr = avgbins(datazbel, datacbel, 0.05, sem=True)
datacbins, datazbins, datacbinserr = avgbins(datazcmb, datac, 0.05, sem=True)
datacabvbins[datacabvbins == 0] = np.nan
datacbelbins[datacbelbins == 0] = np.nan
datacbins[datacbins == 0] = np.nan
# bin x1 with respect to z from dragan
datax1abvbins, datazabvbins, datax1abvbinserr = avgbins(datazabv, datax1abv, 0.05, sem=True)
datax1belbins, datazbelbins, datax1belbinserr = avgbins(datazbel, datax1bel, 0.05, sem=True)
datax1bins, datazbins, datax1binserr = avgbins(datazcmb, datax1, 0.05, sem=True)
datax1abvbins[datax1abvbins == 0] = np.nan
datax1belbins[datax1belbins == 0] = np.nan
datax1bins[datax1bins == 0] = np.nan

# average difference of c bins
coffset = np.subtract(cabvbins[1:6], cbelbins[1:6])
datacoffset = np.subtract(datacabvbins[1:6], datacbelbins[1:6])
x1offset = np.subtract(x1abvbins[1:6], x1belbins[1:6])
datax1offset = np.subtract(datax1abvbins[1:6], datax1belbins[1:6])
muresoffset = np.subtract(muresabvbins[1:6], muresbelbins[1:6])
datamuresoffset = np.subtract(datamuresabvbins[1:6], datamuresbelbins[1:6])

print('Sim mass distribution parameters:')
print('mbar:', mp[0], 'sigma pos:', mp[1], 'sigma neg:', mp[2])
print('Data mass distribution parameters:')
print('mbar:', datamp[0], 'sigma pos:', datamp[1], 'sigma neg:', datamp[2])
print('')

datacoffset = np.subtract(datacabvbins[1:], datacbelbins[1:])
print(datacoffset)
print('')

# print out differences in average values for mass > 10 and mass < 10
print('c differences:')
print('sim:', coffset, 'data:', datacoffset)
print('x1 differences:')
print('sim:', x1offset, 'data:', datax1offset)
print('mures differences:')
print('sim:', muresoffset, 'data:', datamuresoffset)
print('')

cavg = np.average(datac)
ctotal = len(datac)
print('average data c values:')
print('mass < 10:', np.average(datacbel), 'mass > 10:', np.average(datacabv), 'all mass:', cavg)
print('c count above c avg and below c avg for mass > 10')
cup = np.asarray([x >= cavg for x in datacabv])
cdown = np.asarray([x < cavg for x in datacabv])
print(len(datacabv[cup]) / ctotal, len(datacabv[cdown]) / ctotal)
print('c count above c avg and below c avg for mass < 10')
cup = np.asarray([x >= cavg for x in datacbel])
cdown = np.asarray([x < cavg for x in datacbel])
print(len(datacbel[cup]) / ctotal, len(datacbel[cdown]) / ctotal)
callup = np.asarray([x >= cavg for x in datac])
calldown = np.asarray([x < cavg for x in datac])
print('c count above c avg and below c avg for all count')
print(len(datac[callup]) / ctotal, len(datac[calldown]) / ctotal)
print('')

cavg = np.average(c)
ctotal = len(c)
print('average c values:')
print('mass < 10:', np.average(cbel), 'mass > 10:', np.average(cabv), 'all mass:', cavg)
print('normed c count above c avg and below c avg for mass > 10')
cup = np.asarray([x >= cavg for x in cabv])
cdown = np.asarray([x < cavg for x in cabv])
print(len(cabv[cup]) / ctotal, len(cabv[cdown]) / ctotal)
print('normed c count above c avg and below c avg for mass < 10')
cup = np.asarray([x >= cavg for x in cbel])
cdown = np.asarray([x < cavg for x in cbel])
print(len(cbel[cup]) / ctotal, len(cbel[cdown]) / ctotal)
callup = np.asarray([x >= cavg for x in c])
calldown = np.asarray([x < cavg for x in c])
print('normed c count above c avg and below c avg for all mass')
print(len(c[callup]) / ctotal, len(c[calldown]) / ctotal)
print('')

print('sim distribution info:')
p = np.cov(x1, hostmass)
print('x1 correlation:', p[1, 0] / np.sqrt(p[1, 1] * p[0, 0]))
print('x1 cov:', p[1, 0])
gammahat = stats.skew(hostmass)
print('host_logmass skew: ' + str(gammahat))
abvten = np.asarray([x >= 10 for x in hostmass])
belten = np.asarray([x < 10 for x in hostmass])
cabv = c[abvten]
cbel = c[belten]
print(np.mean(cbel), np.mean(cabv))
print('')

print('data distribution info:')
p = np.cov(datax1, datahostmass)
print('x1 correlation:', p[1, 0] / np.sqrt(p[1, 1] * p[0, 0]))
print('x1 cov:', p[1, 0])
gammahat = stats.skew(datahostmass)
print('host_logmass skew: ' + str(gammahat))
abvten = np.asarray([x >= 10 for x in datahostmass])
belten = np.asarray([x < 10 for x in datahostmass])
zabv = zcmb[abvten]
zbel = zcmb[belten]
badzabv = np.asarray([x <= 0.27 for x in zabv])
badzbel = np.asarray([x <= 0.27 for x in zbel])
cabv = datac[abvten]
cbel = datac[belten]
print(np.mean(cbel), np.mean(cabv))
print('')

# plots...
fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 16.0
fig_size[1] = 8.0
os.chdir('/Users/jaredhand/WFIRST_research/SN_distributions/mures_plots/sdss/sim4_mures')

"""
plt.scatter(hostmass, mures, alpha=0.6, label='sim mures')
plt.scatter(datahostmass, datamures, color='magenta', marker='v', alpha=0.6, label='data mures')
plt.scatter(massbins2, muresbins2, color='red', marker='D', alpha=0.8, s=50, label='sim mures bins')
plt.ylabel('mures')
plt.xlabel('hostmass')
plt.ylim(-0.8, 1.)
plt.legend(loc=4)
plt.savefig('hostmass_mures.png')
plt.show()
"""

plt.subplot(121)
plt.hist(mures[aboveten], 25, normed=True, label='mass>10', alpha=0.5)
plt.hist(mures[belowten], 25, normed=True, label='mass<10', alpha=0.5)
plt.xlabel('mures')
plt.title('Sim mures')
plt.xlim(-0.8, 1.)
plt.subplot(122)
plt.hist(datamures[dataaboveten], 25, normed=True, label='mass>10', alpha=0.5)
plt.hist(datamures[databelowten], 25, normed=True, label='mass<10', alpha=0.5)
plt.xlabel('mures')
plt.title('Data mures')
plt.xlim(-0.8, 1.)
plt.legend()
plt.savefig('mures_hists.png')
plt.show()

plt.subplot(121)
plt.errorbar(zabvbins, muresabvbins, yerr=muresabvbinserr,
             color='red', marker='D', alpha=0.7, label='sim>10', fmt='o')
plt.errorbar(zbelbins, muresbelbins, yerr=muresbelbinserr,
             color='blue', marker='s', alpha=0.7, label='sim<10', fmt='o')
plt.xlim(0.05, 0.46)
plt.ylim(-0.25, 0.2)
plt.xlabel('redshift')
plt.ylabel('Hubble residual')
plt.title('Sim')
plt.legend(loc=2)
plt.subplot(122)
plt.errorbar(datazabvbins, datamuresabvbins, yerr=datamuresabvbinserr,
             color='red', marker='D', alpha=0.7, label='data>10', fmt='o')
plt.errorbar(datazbelbins, datamuresbelbins, yerr=datamuresbelbinserr,
             color='blue', marker='s', alpha=0.7, label='data<10', fmt='o')
plt.xlim(0.05, 0.46)
plt.ylim(-0.25, 0.2)
plt.xlabel('redshift')
plt.title('Data')
plt.legend(loc=1)
plt.savefig('z_mures1.png')
plt.show()

plt.errorbar(zbins, muresbins, yerr=muresbelbinserr,
             color='red', marker='D', alpha=0.7, label='sim', fmt='o')
plt.errorbar(datazbins, datamuresbins, yerr=datamuresbelbinserr,
             color='blue', marker='s', alpha=0.7, label='data', fmt='o')
plt.xlim(0.05, 0.46)
plt.ylim(-0.25, 0.2)
plt.xlabel('redshift')
plt.ylabel('hubble residual')
plt.title('All mass')
plt.legend(loc=1)
plt.savefig('z_mures2.png')
plt.show()

plt.errorbar(zabvbins + 0.005, muresabvbins, yerr=muresabvbinserr,
             color='red', marker='D', alpha=0.7, label='sim>10', fmt='o')
plt.errorbar(zbelbins + 0.005, muresbelbins, yerr=muresbelbinserr,
             color='blue', marker='s', alpha=0.7, label='sim<10', fmt='o')
plt.errorbar(datazabvbins - 0.005, datamuresabvbins, yerr=datamuresabvbinserr,
             color='magenta', marker='D', alpha=0.7, label='data>10', fmt='o')
plt.errorbar(datazbelbins - 0.005, datamuresbelbins, yerr=datamuresbelbinserr,
             color='green', marker='s', alpha=0.7, label='data<10', fmt='o')
plt.xlim(0.05, 0.46)
plt.ylim(-0.25, 0.2)
plt.xlabel('redshift')
plt.ylabel('mures')
plt.title('Hubble Residual vs Redshift')
plt.legend(loc=1)
plt.savefig('z_mures3.png')
plt.show()

plt.subplot(121)
plt.errorbar(zabvbins, cabvbins, yerr=cabvbinserr,
             color='red', marker='D', alpha=0.7, label='sim>10', fmt='o')
plt.errorbar(zbelbins, cbelbins, yerr=cbelbinserr,
             color='blue', marker='s', alpha=0.7, label='sim<10', fmt='o')
plt.xlim(0.05, 0.46)
plt.xlabel('redshift')
plt.ylabel('c')
plt.title('Sim')
plt.legend(loc=1)
plt.ylim(-0.2, 0.2)
plt.subplot(122)
plt.errorbar(datazabvbins, datacabvbins, yerr=datacabvbinserr,
             color='red', marker='D', alpha=0.7, label='data>10', fmt='o')
plt.errorbar(datazbelbins, datacbelbins, yerr=datacbelbinserr,
             color='blue', marker='s', alpha=0.7, label='data<10', fmt='o')
plt.xlim(0.05, 0.46)
plt.xlabel('redshift')
plt.title('Data')
plt.legend(loc=1)
plt.ylim(-0.2, 0.2)
plt.savefig('z_c1.png')
plt.show()

plt.errorbar(zbins, cbins, yerr=cbinserr,
             color='red', marker='D', label='sim', fmt='o')
plt.errorbar(datazbins, datacbins, yerr=datacbinserr,
             color='blue', marker='s', label='data', fmt='o')
plt.xlim(0.05, 0.46)
plt.xlabel('redshift')
plt.ylabel('c')
plt.title('All mass')
plt.legend(loc=1)
plt.ylim(-0.2, 0.2)
plt.savefig('z_c2.png')
plt.show()

plt.errorbar(zabvbins + 0.005, cabvbins, yerr=cabvbinserr,
             color='red', marker='D', alpha=0.7, label='sim>10', fmt='o')
plt.errorbar(zbelbins + 0.005, cbelbins, yerr=cbelbinserr,
             color='blue', marker='s', alpha=0.7, label='sim<10', fmt='o')
plt.errorbar(datazabvbins - 0.005, datacabvbins, yerr=datacabvbinserr,
             color='magenta', marker='D', alpha=0.7, label='data>10', fmt='o')
plt.errorbar(datazbelbins - 0.005, datacbelbins, yerr=datacbelbinserr,
             color='green', marker='s', alpha=0.7, label='data<10', fmt='o')
plt.xlim(0.05, 0.46)
plt.xlabel('redshift')
plt.ylabel('c')
plt.title('Color vs Redshift')
plt.legend(loc=1)
plt.ylim(-0.2, 0.2)
plt.savefig('z_c3.png')
plt.show()

plt.subplot(121)
plt.errorbar(zabvbins, x1abvbins, yerr=x1abvbinserr,
             color='red', marker='D', alpha=0.7, label='sim>10', fmt='o')
plt.errorbar(zbelbins, x1belbins, yerr=x1belbinserr,
             color='blue', marker='s', alpha=0.7, label='sim<10', fmt='o')
plt.xlim(0.05, 0.46)
plt.ylim(-1.5, 1.5)
plt.xlabel('redshift')
plt.ylabel('x1')
plt.title('Sim')
plt.legend(loc=4)
plt.subplot(122)
plt.errorbar(datazabvbins, datax1abvbins, yerr=datax1abvbinserr,
             color='red', marker='D', alpha=0.7, label='data>10', fmt='o')
plt.errorbar(datazbelbins, datax1belbins, yerr=datax1belbinserr,
             color='blue', marker='s', alpha=0.7, label='data<10', fmt='o')
plt.xlim(0.05, 0.46)
plt.ylim(-1.5, 1.5)
plt.xlabel('redshift')
plt.title('Data')
plt.legend(loc=4)
plt.savefig('z_x11.png')
plt.show()

plt.errorbar(zbins, x1bins, yerr=x1binserr,
             color='red', marker='D', alpha=0.7, label='sim', fmt='o')
plt.errorbar(datazbins, datax1bins, datax1binserr,
             color='blue', marker='s', alpha=0.7, label='data', fmt='o')
plt.xlim(0.05, 0.46)
plt.ylim(-1.5, 1.5)
plt.xlabel('redshift')
plt.ylabel('x1')
plt.title('All mass')
plt.legend(loc=4)
plt.savefig('z_x12.png')
plt.show()

plt.errorbar(zabvbins + 0.005, x1abvbins, yerr=x1abvbinserr,
             color='red', marker='D', alpha=0.7, label='sim>10', fmt='o')
plt.errorbar(zbelbins + 0.005, x1belbins, yerr=x1belbinserr,
             color='blue', marker='s', alpha=0.7, label='sim<10', fmt='o')
plt.errorbar(datazabvbins - 0.005, datax1abvbins, yerr=datax1abvbinserr,
             color='magenta', marker='D', alpha=0.7, label='data>10', fmt='o')
plt.errorbar(datazbelbins - 0.005, datax1belbins, yerr=datax1belbinserr,
             color='green', marker='s', alpha=0.7, label='data<10', fmt='o')
plt.xlim(0.05, 0.46)
plt.ylim(-1.5, 1.5)
plt.xlabel('redshift')
plt.ylabel('x1')
plt.title('Stretch vs Redshift')
plt.legend(loc=1)
plt.savefig('z_x13.png')
plt.show()

"""
plt.scatter(c, mures, alpha=0.5)
plt.scatter(datac, datamures, alpha=0.6, marker='v', color='red')
plt.xlabel('c')
plt.ylabel('mures')
plt.show()

plt.scatter(x1, mures, alpha=0.5)
plt.scatter(datax1, datamures, alpha=0.6, marker='v', color='red')
plt.xlabel('x1')
plt.ylabel('mures')
plt.show()
"""
plt.scatter(hostmass, x1, alpha=0.5)
plt.scatter(datahostmass, datax1, alpha=0.6, marker='v', color='red')
plt.xlabel('hostmass')
plt.ylabel('x1')
plt.show()


# heat maps
x1bounds = [7., 13., -4., 3.]
cbounds = [7., 13., -0.3, 0.3]
massrange = np.arange(x1bounds[0], x1bounds[1], 1.)
cmassrange = np.arange(x1bounds[0], x1bounds[1], 0.5)
x1range = np.arange(x1bounds[2], x1bounds[3], 1.)
crange = np.arange(cbounds[2], cbounds[3], 0.05)
crange[np.abs(crange) < 1e-10] = 0.

plt.title('color')
plt.subplot(121)
plt.pcolormesh(cmassmap)
plt.xticks(np.arange(len(cmassrange)), cmassrange)
plt.yticks(np.arange(len(crange)), crange)
plt.title('simulation')
plt.xlabel('host mass')
plt.ylabel('c')
plt.colorbar()
plt.subplot(122)
plt.pcolormesh(datacmassmap)
plt.xticks(np.arange(len(cmassrange)), cmassrange)
plt.yticks(np.arange(len(crange)), crange)
plt.title('data')
plt.xlabel('host mass')
plt.colorbar()
plt.savefig('cheatmaps.png')
plt.show()


plt.subplot(121)
plt.pcolormesh(x1massmap)
plt.xticks(np.arange(len(massrange)), massrange)
plt.yticks(np.arange(len(x1range)), x1range)
plt.title('simulation')
plt.xlabel('host mass')
plt.ylabel('x1')
plt.colorbar()
plt.subplot(122)
plt.pcolormesh(datax1massmap)
plt.xticks(np.arange(len(massrange)), massrange)
plt.yticks(np.arange(len(x1range)), x1range)
plt.title('data')
plt.xlabel('host mass')
plt.colorbar()
plt.savefig('x1heatmaps.png')
plt.show()


# plt.subplot(311)
plt.scatter(zbins[1:6], muresoffset, color='red', marker='v', alpha=0.6, label='sim', s=50)
plt.scatter(datazbins[1:6], datamuresoffset, marker='D', alpha=0.6, label='data', s=50)
plt.legend()
plt.title('Mures Difference vs Redshift')
plt.show()


"""
plt.subplot(312)
plt.scatter(zbins[1:6], coffset, color='red', marker='v', alpha=0.6, label='sim')
plt.scatter(datazbins[1:6], datacoffset, marker='D', alpha=0.6, label='data')
plt.legend()
plt.title('Color Difference vs Redshift')
plt.subplot(313)
plt.scatter(zbins[1:6], x1offset, color='red', marker='v', alpha=0.6, label='sim')
plt.scatter(datazbins[1:6], x1offset, marker='D', alpha=0.6, label='data')
plt.legend()
plt.title('Stretch Difference vs Redshift')
plt.show()
"""