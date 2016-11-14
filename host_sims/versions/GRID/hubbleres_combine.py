import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import decimal
from config import *

# import dragan data
datafitres = 'dragan2.fitres'
os.chdir(homedir + '/dragan')
with open(datafitres, 'r') as f:
    for line in f:
        if 'VARNAME' in line:
            datavars = line
            break
with open(datafitres, 'r') as f:
    for line in f:
        if 'NVAR' in line:
            datavarscount = int(line.split(' ')[1]) - 2
            break
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
        datamures0 = np.copy(dragan[:, i])

newdragan = np.copy(dragan)

# only use z<0.1 unless cid=1,4,15
bad_z = np.asarray([x <= 0.1 for x in datazcmb])
bad_cidint = np.asarray([(x == 1 or x == 4) for x in datacidint])
bad_zandcid = np.logical_or(bad_z, bad_cidint)
datahostmass = datahostmass[bad_zandcid]
datazcmb = datazcmb[bad_zandcid]
datax1 = datax1[bad_zandcid]
datac = datac[bad_zandcid]
datamures0 = datamures0[bad_zandcid]
newdragan = newdragan[bad_zandcid]
datacidint = datacidint[bad_zandcid]

# remove host_logmass < 7 from dragan data.
bad_logmass = np.asarray([x >= 7. for x in datahostmass])
datahostmass = datahostmass[bad_logmass]
datazcmb = datazcmb[bad_logmass]
datax1 = datax1[bad_logmass]
datac = datac[bad_logmass]
datamures0 = datamures0[bad_logmass]
datacidint = datacidint[bad_logmass]
newdragan = newdragan[bad_logmass]

# remove mures > abs(10) from dragan data
bad_mures = np.asarray([x < 10 for x in np.abs(datamures0)])
datahostmass = datahostmass[bad_mures]
datazcmb = datazcmb[bad_mures]
datax1 = datax1[bad_mures]
datac = datac[bad_mures]
datamures0 = datamures0[bad_mures]
datacidint = datacidint[bad_mures]
newdragan = newdragan[bad_mures]

# import sim data
fitres = 'composite.fitres'
os.chdir(homedir + '/fitres/composite')
with open(fitres, 'r') as f:
    for line in f:
        if 'VARNAME' in line:
            var = line
            break
with open(fitres, 'r') as f:
    for line in f:
        if 'NVAR' in line:
            varscount = int(line.split(' ')[1]) - 2
            break
var = var.split()
var.remove('VARNAMES:')
# vars.remove('FIELD')
columns = tuple(list(range(1, 52)))
data = np.loadtxt(fitres, dtype=float, skiprows=2, usecols=columns)
for i in range(varscount):
    if var[i] == 'MURES':
        mures = np.copy(data[:, i])
    elif var[i] == 'HOST_LOGMASS':
        hostmass = np.copy(data[:, i])
    elif var[i] == 'zCMB':
        zcmb = np.copy(data[:, i])
    elif var[i] == 'x1':
        x1 = np.copy(data[:, i])
    elif var[i] == 'c':
        c = np.copy(data[:, i])
    elif var[i] == 'FITPROB':
        fitprob = np.copy(data[:, i])

# print(max(mures), min(mures))
# print(max(c), min(c))
# print(max(x1), min(x1))

bad_c = np.asarray([x < 0.3 for x in np.abs(c)])
c = c[bad_c]
x1 = x1[bad_c]
zcmb = zcmb[bad_c]
mures = mures[bad_c]
hostmass = hostmass[bad_c]
fitprob = fitprob[bad_c]

bad_x1 = np.asarray([x < 4. for x in np.abs(x1)])
c = c[bad_x1]
x1 = x1[bad_x1]
zcmb = zcmb[bad_x1]
mures = mures[bad_x1]
hostmass = hostmass[bad_x1]
fitprob = fitprob[bad_x1]

bad_mures = np.asarray([x < 1. for x in np.abs(mures)])
c = c[bad_mures]
x1 = x1[bad_mures]
zcmb = zcmb[bad_mures]
mures = mures[bad_mures]
hostmass = hostmass[bad_mures]
fitprob = fitprob[bad_mures]

bad_fitprob = np.asarray([x > 0.01 for x in np.abs(fitprob)])
c = c[bad_fitprob]
x1 = x1[bad_fitprob]
zcmb = zcmb[bad_fitprob]
mures = mures[bad_fitprob]
hostmass = hostmass[bad_fitprob]
fitprob = fitprob[bad_fitprob]

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
dataaboveavg = np.median(datamures0[dataaboveten])
databelowavg = np.median(datamures0[databelowten])
datamuresavg = np.median(datamures0)
print('DATA')
print('Mass greater than 10:', dataaboveavg)
print('Mass less than 10:', databelowavg)
print('median:', datamuresavg)
print('std:', np.std(datamures0), 'std above:', np.std(datamures0[dataaboveten]),
      'std below:', np.std(datamures0[databelowten]))
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
hostmassabv = np.asarray(hostmass[aboveten])
hostmassbel = np.asarray(hostmass[belowten])

# split up data x1, c and zcmb based on mass greater or less than 10
datacabv = np.asarray(datac[dataaboveten])
datacbel = np.asarray(datac[databelowten])
datax1abv = np.asarray(datax1[dataaboveten])
datax1bel = np.asarray(datax1[databelowten])
datazabv = np.asarray(datazcmb[dataaboveten])
datazbel = np.asarray(datazcmb[databelowten])


def mures_mixer(arr, idcol, zcol, iddict, zmurescol):
    """
    Uses input idlist and clist to return correct mures from data based on ID survey value
    :param zmurescol: mures column for lowz
    :param zcol: zcmb column
    :param idcol: redshift
    :param arr: data array
    :param iddict: dict of ID keys and mures column values from surveys
    :return: array of mures values corresponding to the correct ID and redshift.
    """
    newmures = []
    failcount = 0
    for i in arr:
        id = int(i[idcol])
        if id in list(iddict.keys()) and i[zcol] > 0.1:
            mcol = iddict[id]
            newmures.append(i[mcol])
        elif i[zcol] <= 0.1:
            newmures.append(i[zmurescol])
        else:
            print(i[idcol], i[zcol])
            failcount += 1
    print(failcount)
    return np.asarray(newmures)


def mass_params(massarr, zeroed=False):
    #### NOTE: I don't think this works... ####
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
# mp = mass_params(hostmass)
# datamp = mass_params(datahostmass)


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


def digitize(arr1, arr2, bounds, step1=1., step2=1., norm=False):
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


def diff_calc(arrxabv, arrxbel, arryabv, arrybel, xcutmin, xcutmax):
    """
    arrxabv, arryabv correspond to arrays for mass above 10 and arrxbel and arrybel correspond to arrays for mass
    below ten.  This function removes arrxabv and arrxbel x values that are outside of domain specified via
    xcutmin and xcutmax.  The offset for each x bin is calculated for the above mass 10 arrays and the below mass 10
    arrays.
    :param arrxabv: x-axis with mass above 10
    :param arrxbel: x-axis with mass below 10
    :param arryabv: y-axis with mass above 10
    :param arrybel: y-axis with mass below 10
    :param xcutmin: beginning of x domain
    :param xcutmax: end of x domain
    :return: Concatenated arryabv subtracted by concatenated arrybel
    """
    badabv = np.asarray([xcutmax >= x > xcutmin for x in arrxabv])
    badbel = np.asarray([xcutmax >= x > xcutmin for x in arrxbel])
    narryabv = arryabv[badabv]
    narrybel = arrybel[badbel]
    return np.subtract(narryabv, narrybel)

# grab correct mures values from dragan
datadict = {1: 34, 4: 37, 15: 37}  # this was found via guess and check...
datamures = mures_mixer(newdragan, dataidcol, datazcol, datadict, 37)
# split up mures into mass >= 10 and mass < 10 subsets
datamuresabv = np.asarray(datamures[dataaboveten])
datamuresbel = np.asarray(datamures[databelowten])

# make cells for heat map
massx1bounds = [7., 13., -4., 3.]
masscbounds = [7., 13., -0.3, 0.3]
x1massmap = digitize(hostmass, x1, massx1bounds)
cmassmap = digitize(hostmass, c, masscbounds, step1=0.5, step2=0.05)
datax1massmap = digitize(datahostmass, datax1, massx1bounds)
datacmassmap = digitize(datahostmass, datac, masscbounds, step1=0.5, step2=0.05)

# zbin size
z = 0.1
# bin mures with respect to mass
muresbins2, massbins2, massbinserr2 = avgbins(hostmass, mures, 1, sem=True)
# bin mures with respect to z
muresabvbins, zabvbins, muresabvbinserr = avgbins(zabv, muresabv, z, sem=True)
muresbelbins, zbelbins, muresbelbinserr = avgbins(zbel, muresbel, z, sem=True)
muresbins, zbins, muresbinserr = avgbins(zcmb, mures, z, sem=True)
muresabvbins[muresabvbins == 0] = np.nan
muresbelbins[muresbelbins == 0] = np.nan
muresbins[muresbins == 0] = np.nan
muresabvbinserr[muresabvbinserr == 0] = np.nan
muresbelbinserr[muresbelbinserr == 0] = np.nan
muresbinserr[muresbinserr == 0] = np.nan
# bin c with respect to z
cabvbins, zabvbins, cabvbinserr = avgbins(zabv, cabv, z, sem=True)
cbelbins, zbelbins, cbelbinserr = avgbins(zbel, cbel, z, sem=True)
cbins, zbins, cbinserr = avgbins(zcmb, c, z, sem=True)
cabvbins[cabvbins == 0] = np.nan
cbelbins[cbelbins == 0] = np.nan
cbins[cbins == 0] = np.nan
cabvbinserr[cabvbinserr == 0] = np.nan
cbelbinserr[cbelbinserr == 0] = np.nan
cbinserr[cbinserr == 0] = np.nan
# bin x1 with respect to z
x1abvbins, zabvbins, x1abvbinserr = avgbins(zabv, x1abv, z, sem=True)
x1belbins, zbelbins, x1belbinserr = avgbins(zbel, x1bel, z, sem=True)
x1bins, zbins, x1binserr = avgbins(zcmb, x1, z, sem=True)
x1abvbins[x1abvbins == 0] = np.nan
x1belbins[x1belbins == 0] = np.nan
x1bins[x1bins == 0] = np.nan
x1abvbinserr[x1abvbinserr == 0] = np.nan
x1belbinserr[x1belbinserr == 0] = np.nan
x1binserr[x1binserr == 0] = np.nan

# bin mures with respect to mass from dragan
datamuresbins2, datamassbins2, datamassbinserr2 = avgbins(datahostmass, datamures, 1, sem=True)
# bin mures with respect to z from dragan
datamuresabvbins, datazabvbins, datamuresabvbinserr = avgbins(datazabv, datamuresabv, z, sem=True)
datamuresbelbins, datazbelbins, datamuresbelbinserr = avgbins(datazbel, datamuresbel, z, sem=True)
datamuresbins, datazbins, datamuresbinserr = avgbins(datazcmb, datamures, z, sem=True)
datamuresabvbins[datamuresabvbins == 0] = np.nan
datamuresbelbins[datamuresbelbins == 0] = np.nan
datamuresbins[datamuresbins == 0] = np.nan
datamuresabvbinserr[datamuresabvbinserr == 0] = np.nan
datamuresbelbinserr[datamuresbelbinserr == 0] = np.nan
datamuresbinserr[datamuresbinserr == 0] = np.nan
# bin c with respect to z from dragan
datacabvbins, datazabvbins, datacabvbinserr = avgbins(datazabv, datacabv, z, sem=True)
datacbelbins, datazbelbins, datacbelbinserr = avgbins(datazbel, datacbel, z, sem=True)
datacbins, datazbins, datacbinserr = avgbins(datazcmb, datac, z, sem=True)
datacabvbins[datacabvbins == 0] = np.nan
datacbelbins[datacbelbins == 0] = np.nan
datacbins[datacbins == 0] = np.nan
datacabvbinserr[datacabvbinserr == 0] = np.nan
datacbelbinserr[datacbelbinserr == 0] = np.nan
datacbinserr[datacbinserr == 0] = np.nan
# bin x1 with respect to z from dragan
datax1abvbins, datazabvbins, datax1abvbinserr = avgbins(datazabv, datax1abv, z, sem=True)
datax1belbins, datazbelbins, datax1belbinserr = avgbins(datazbel, datax1bel, z, sem=True)
datax1bins, datazbins, datax1binserr = avgbins(datazcmb, datax1, z, sem=True)
datax1abvbins[datax1abvbins == 0] = np.nan
datax1belbins[datax1belbins == 0] = np.nan
datax1bins[datax1bins == 0] = np.nan
datax1abvbinserr[datax1abvbinserr == 0] = np.nan
datax1belbinserr[datax1belbinserr == 0] = np.nan
datax1binserr[datax1binserr == 0] = np.nan

# average difference of data bins with respect to redshift
coffset = diff_calc(zabvbins, zbelbins, cabvbins, cbelbins, 0., 1.)
datacoffset = diff_calc(datazabvbins, datazbelbins, datacabvbins, datacbelbins, 0., 1.)
x1offset = diff_calc(zabvbins, zbelbins, x1abvbins, x1belbins, 0., 1.)
datax1offset = diff_calc(datazabvbins, datazbelbins, datax1abvbins, datax1belbins, 0., 1.)
muresoffset = diff_calc(zabvbins, zbelbins, muresabvbins, muresbelbins, 0., 1.)
datamuresoffset = diff_calc(datazabvbins, datazbelbins, datamuresabvbins, datamuresbelbins, 0., 1.)

# global offset avgs
cabvavg, cbelavg = np.nanmean(cabv), np.nanmean(cbel)
cgoffset = cabvavg - cbelavg
x1abvavg, x1belavg = np.nanmean(x1abv), np.nanmean(x1bel)
x1goffist = x1abvavg - x1belavg
muresabvavg, muresbelavg = np.nanmean(muresabv), np.nanmean(muresbel)
muresgoffset = muresabvavg - muresbelavg
datacabvavg, datacbelavg = np.nanmean(datacabv), np.nanmean(datacbel)
datacgoffset = datacabvavg - datacbelavg
datax1abvavg, datax1belavg = np.nanmean(datax1abv), np.nanmean(datax1bel)
datax1goffist = datax1abvavg - datax1belavg
datamuresabvavg, datamuresbelavg = np.nanmean(datamuresabv), np.nanmean(datamuresbel)
datamuresgoffset = datamuresabvavg - datamuresbelavg
print('Mass Step:', muresgoffset)
"""
print('Sim mass distribution parameters:')
print('mbar:', mp[0], 'sigma pos:', mp[1], 'sigma neg:', mp[2])
print('Data mass distribution parameters:')
print('mbar:', datamp[0], 'sigma pos:', datamp[1], 'sigma neg:', datamp[2])
print('')
"""
# print out differences in average values for mass > 10 and mass < 10
print('c differences:')
print('sim:', coffset, 'data:', datacoffset)
print('x1 differences:')
print('sim:', x1offset, 'data:', datax1offset)
print('mures differences:')
print('sim:', muresoffset, 'data:', datamuresoffset)
print('')

datacoffset = np.subtract(datacabvbins[1:], datacbelbins[1:])
print(datacoffset)
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
zaxis = np.arange(0., 1., 0.1)

# set decimal precision for plot annotations
decimal.getcontext().prec = 2
os.chdir(homedir + '/mures_plots/composite')

# mures histogram plots
plt.subplot(121)
plt.hist(mures[aboveten], 50, normed=True, label='mass>10', alpha=0.5)
plt.hist(mures[belowten], 50, normed=True, label='mass<10', alpha=0.5)
plt.xlabel('mures')
plt.title('Sim mures')
plt.xlim(-0.75, 0.75)
plt.subplot(122)
plt.hist(datamures[dataaboveten], 25, normed=True, label='mass>10', alpha=0.5)
plt.hist(datamures[databelowten], 25, normed=True, label='mass<10', alpha=0.5)
plt.xlabel('mures')
plt.title('Data mures')
plt.xlim(-0.75, 0.75)
plt.legend()
plt.savefig('mures_hists.png')
plt.show()

# y-axis bounds for plots
myb = [-0.5, 0.5]
cyb = [-0.2, 0.2]
x1yb = [-1.5, 1.5]

# main error bar plots
plt.subplot(121)
plt.errorbar(zabvbins, muresabvbins, yerr=muresabvbinserr,
             color='red', marker='D', alpha=0.7, label='sim>10', fmt='o')
plt.errorbar(zbelbins, muresbelbins, yerr=muresbelbinserr,
             color='blue', marker='s', alpha=0.7, label='sim<10', fmt='o')
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis)
plt.ylim(myb[0], myb[1])
plt.xlabel('redshift')
plt.ylabel('Hubble residual')
plt.title('Sim')
plt.legend(loc=1)
plt.text(0.02, myb[1] * 0.9, 'Offset Mean: ' + str(decimal.Decimal(muresgoffset) + decimal.Decimal(0)),
         fontsize=16)
"""
for i in range(len(muresabvbins)):
    if muresabvbins[i] < muresbelbins[i]:
        plt.annotate(str(decimal.Decimal(muresoffset[i]) + decimal.Decimal(0)),
                     xy=(zabvbins[i], muresabvbins[i] - muresoffset[i] / 2),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.5),
                     xycoords='data', xytext=(zabvbins[i] + 0.02, muresabvbins[i] + 0.4))
    else:
        plt.annotate(str(decimal.Decimal(muresoffset[i]) + decimal.Decimal(0)),
                     xy=(zabvbins[i], muresbelbins[i] + muresoffset[i] / 2),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.5),
                     xycoords='data', xytext=(zabvbins[i] + 0.02, muresabvbins[i] + 0.4))
"""
plt.subplot(122)
plt.errorbar(datazabvbins, datamuresabvbins, yerr=datamuresabvbinserr,
             color='red', marker='D', alpha=0.7, label='data>10', fmt='o')
plt.errorbar(datazbelbins, datamuresbelbins, yerr=datamuresbelbinserr,
             color='blue', marker='s', alpha=0.7, label='data<10', fmt='o')
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis)
plt.ylim(myb[0], myb[1])
plt.xlabel('redshift')
plt.title('Data')
plt.legend(loc=1)
plt.text(0.02, myb[1] * 0.9, 'Offset Mean: ' + str(decimal.Decimal(datamuresgoffset) + decimal.Decimal(0)),
         fontsize=16)
"""
for i in range(len(datamuresabvbins)):
    if datamuresabvbins[i] < datamuresbelbins[i]:
        plt.annotate(str(decimal.Decimal(datamuresoffset[i]) + decimal.Decimal(0)),
                     xy=(datazabvbins[i], datamuresabvbins[i] - datamuresoffset[i] / 2),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.5),
                     xycoords='data', xytext=(datazabvbins[i] + 0.02, datamuresabvbins[i] + 0.4))
    else:
        plt.annotate(str(decimal.Decimal(datamuresoffset[i]) + decimal.Decimal(0)),
                     xy=(datazabvbins[i], datamuresbelbins[i] + datamuresoffset[i] / 2),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.5),
                     xycoords='data', xytext=(datazabvbins[i] + 0.02, datamuresabvbins[i] + 0.4))
"""
plt.savefig('z_mures1.png')
plt.show()

plt.errorbar(zbins, muresbins, yerr=muresbelbinserr,
             color='red', marker='D', alpha=0.7, label='sim', fmt='o')
plt.errorbar(datazbins, datamuresbins, yerr=datamuresbelbinserr,
             color='blue', marker='s', alpha=0.7, label='data', fmt='o')
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis)
plt.ylim(myb[0], myb[1])
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
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis)
plt.ylim(myb[0], myb[1])
plt.xlabel('redshift')
plt.ylabel('mures')
plt.title('Hubble Residual vs Redshift')
plt.legend(loc=1)
plt.text(0.02, myb[1] * 0.9, 'Sim Offset Mean: ' + str(decimal.Decimal(muresgoffset) + decimal.Decimal(0)),
         fontsize=16)
plt.text(0.02, myb[1] * 0.75, 'Obs Offset Mean: ' + str(decimal.Decimal(datamuresgoffset) + decimal.Decimal(0)),
         fontsize=16)
plt.savefig('z_mures3.png')
plt.show()

plt.errorbar(zabvbins + 0.005, muresabvbins, yerr=muresabvbinserr,
             color='red', marker='D', alpha=0.7, label='sim>10', fmt='o')
plt.errorbar(zbelbins + 0.005, muresbelbins, yerr=muresbelbinserr,
             color='blue', marker='s', alpha=0.7, label='sim<10', fmt='o')
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis)
plt.ylim(myb[0], myb[1])
plt.xlabel('redshift')
plt.ylabel('mures')
plt.title('Hubble Residual vs Redshift')
plt.legend(loc=1)
plt.text(0.02, myb[1] * 0.9, 'Sim Offset Mean: ' + str(decimal.Decimal(muresgoffset) + decimal.Decimal(0)),
         fontsize=16)
plt.savefig('z_mures4.png')
plt.show()

plt.subplot(121)
plt.errorbar(zabvbins, cabvbins, yerr=cabvbinserr,
             color='red', marker='D', alpha=0.7, label='sim>10', fmt='o')
plt.errorbar(zbelbins, cbelbins, yerr=cbelbinserr,
             color='blue', marker='s', alpha=0.7, label='sim<10', fmt='o')
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis)
plt.xlabel('redshift')
plt.ylabel('c')
plt.title('Sim')
plt.legend(loc=1)
plt.ylim(cyb[0], cyb[1])
plt.text(0.02, cyb[1] * 0.9, 'Offset Mean: ' + str(decimal.Decimal(cgoffset) + decimal.Decimal(0)),
         fontsize=16)
"""
for i in range(len(cabvbins)):
    if cabvbins[i] < cbelbins[i]:
        plt.annotate(str(decimal.Decimal(coffset[i]) + decimal.Decimal(0)),
                     xy=(zabvbins[i], cabvbins[i] - coffset[i] / 2),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.5),
                     xycoords='data', xytext=(zabvbins[i] + 0.02, cabvbins[i] + 0.05))
    else:
        plt.annotate(str(decimal.Decimal(coffset[i]) + decimal.Decimal(0)),
                     xy=(zabvbins[i], cbelbins[i] + coffset[i] / 2),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.5),
                     xycoords='data', xytext=(zabvbins[i] + 0.02, cabvbins[i] + 0.05))
"""
plt.subplot(122)
plt.errorbar(datazabvbins, datacabvbins, yerr=datacabvbinserr,
             color='red', marker='D', alpha=0.7, label='data>10', fmt='o')
plt.errorbar(datazbelbins, datacbelbins, yerr=datacbelbinserr,
             color='blue', marker='s', alpha=0.7, label='data<10', fmt='o')
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis)
plt.xlabel('redshift')
plt.title('Data')
plt.legend(loc=1)
plt.ylim(cyb[0], cyb[1])
plt.text(0.02, cyb[1] * 0.9, 'Offset Mean: ' + str(decimal.Decimal(datacgoffset) + decimal.Decimal(0)),
         fontsize=16)
"""
for i in range(len(datacoffset)):
    if datacabvbins[i] < datacbelbins[i]:
        plt.annotate(str(decimal.Decimal(datacoffset[i]) + decimal.Decimal(0)),
                     xy=(datazabvbins[i], datacabvbins[i] - datacoffset[i] / 2),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.5),
                     xycoords='data', xytext=(datazabvbins[i] + 0.02, datacabvbins[i] + 0.05))
    else:
        plt.annotate(str(decimal.Decimal(datacoffset[i]) + decimal.Decimal(0)),
                     xy=(datazabvbins[i], datacbelbins[i] + datacoffset[i] / 2),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.5),
                     xycoords='data', xytext=(datazabvbins[i] + 0.02, datacabvbins[i] + 0.05))
"""
plt.savefig('z_c1.png')
plt.show()

plt.errorbar(zbins, cbins, yerr=cbinserr,
             color='red', marker='D', label='sim', fmt='o')
plt.errorbar(datazbins, datacbins, yerr=datacbinserr,
             color='blue', marker='s', label='data', fmt='o')
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis)
plt.xlabel('redshift')
plt.ylabel('c')
plt.title('All mass')
plt.legend(loc=1)
plt.ylim(cyb[0], cyb[1])
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
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis)
plt.xlabel('redshift')
plt.ylabel('c')
plt.title('Color vs Redshift')
plt.legend(loc=1)
plt.ylim(cyb[0], cyb[1])
plt.text(0.02, cyb[1] * 0.9, 'Sim Offset Mean: ' + str(decimal.Decimal(cgoffset) + decimal.Decimal(0)),
         fontsize=16)
plt.text(0.02, cyb[1] * 0.75, 'Obs Offset Mean: ' + str(decimal.Decimal(datacgoffset) + decimal.Decimal(0)),
         fontsize=16)
plt.savefig('z_c3.png')
plt.show()

plt.errorbar(zabvbins + 0.005, cabvbins, yerr=cabvbinserr,
             color='red', marker='D', alpha=0.7, label='sim>10', fmt='o')
plt.errorbar(zbelbins + 0.005, cbelbins, yerr=cbelbinserr,
             color='blue', marker='s', alpha=0.7, label='sim<10', fmt='o')
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis)
plt.xlabel('redshift')
plt.ylabel('c')
plt.title('Color vs Redshift')
plt.legend(loc=1)
plt.ylim(cyb[0], cyb[1])
plt.text(0.02, cyb[1] * 0.9, 'Sim Offset Mean: ' + str(decimal.Decimal(cgoffset) + decimal.Decimal(0)),
         fontsize=16)
plt.savefig('z_c4.png')
plt.show()

plt.subplot(121)
plt.errorbar(zabvbins, x1abvbins, yerr=x1abvbinserr,
             color='red', marker='D', alpha=0.7, label='sim>10', fmt='o')
plt.errorbar(zbelbins, x1belbins, yerr=x1belbinserr,
             color='blue', marker='s', alpha=0.7, label='sim<10', fmt='o')
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis)
plt.ylim(x1yb[0], x1yb[1])
plt.xlabel('redshift')
plt.ylabel('x1')
plt.title('Sim')
plt.legend(loc=4)
plt.text(0.02, x1yb[1] * 0.9, 'Offset Mean: ' + str(decimal.Decimal(x1goffist) + decimal.Decimal(0)),
         fontsize=16)
"""
for i in range(len(x1offset)):
    if x1abvbins[i] < x1belbins[i]:
        plt.annotate(str(decimal.Decimal(x1offset[i]) + decimal.Decimal(0)),
                     xy=(zabvbins[i], x1abvbins[i] - x1offset[i] / 2),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.5),
                     xycoords='data', xytext=(zabvbins[i] + 0.02, x1abvbins[i] + 0.7))
    else:
        plt.annotate(str(decimal.Decimal(x1offset[i]) + decimal.Decimal(0)),
                     xy=(zabvbins[i], x1belbins[i] + x1offset[i] / 2),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.5),
                     xycoords='data', xytext=(x1abvbins[i] + 0.02, x1abvbins[i] + 0.7))
"""
plt.subplot(122)
plt.errorbar(datazabvbins, datax1abvbins, yerr=datax1abvbinserr,
             color='red', marker='D', alpha=0.7, label='data>10', fmt='o')
plt.errorbar(datazbelbins, datax1belbins, yerr=datax1belbinserr,
             color='blue', marker='s', alpha=0.7, label='data<10', fmt='o')
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis)
plt.ylim(x1yb[0], x1yb[1])
plt.xlabel('redshift')
plt.title('Data')
plt.legend(loc=4)
plt.text(0.02, x1yb[1] * 0.9, 'Offset Mean: ' + str(decimal.Decimal(datax1goffist) + decimal.Decimal(0)),
         fontsize=16)
"""
for i in range(len(datax1offset)):
    if datax1abvbins[i] < datax1belbins[i]:
        plt.annotate(str(decimal.Decimal(datax1offset[i]) + decimal.Decimal(0)),
                     xy=(datazabvbins[i], datax1abvbins[i] - datax1offset[i] / 2),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.5),
                     xycoords='data', xytext=(datazabvbins[i] + 0.02, datax1abvbins[i] + 0.7))
    else:
        plt.annotate(str(decimal.Decimal(datax1offset[i]) + decimal.Decimal(0)),
                     xy=(datazabvbins[i], datax1belbins[i] + datax1offset[i] / 2),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.5),
                     xycoords='data', xytext=(datax1abvbins[i] + 0.02, datax1abvbins[i] + 0.7))
"""
plt.savefig('z_x11.png')
plt.show()

plt.errorbar(zbins, x1bins, yerr=x1binserr,
             color='red', marker='D', alpha=0.7, label='sim', fmt='o')
plt.errorbar(datazbins, datax1bins, datax1binserr,
             color='blue', marker='s', alpha=0.7, label='data', fmt='o')
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis)
plt.ylim(x1yb[0], x1yb[1])
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
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis)
plt.ylim(x1yb[0], x1yb[1])
plt.xlabel('redshift')
plt.ylabel('x1')
plt.title('Stretch vs Redshift')
plt.legend(loc=1)
plt.text(0.02, x1yb[1] * 0.9, 'Sim Offset Mean: ' + str(decimal.Decimal(x1goffist) + decimal.Decimal(0)),
         fontsize=16)
plt.text(0.02, x1yb[1] * 0.75, 'Obs Offset Mean: ' + str(decimal.Decimal(datax1goffist) + decimal.Decimal(0)),
         fontsize=16)
plt.savefig('z_x13.png')
plt.show()

plt.errorbar(zabvbins + 0.005, x1abvbins, yerr=x1abvbinserr,
             color='red', marker='D', alpha=0.7, label='sim>10', fmt='o')
plt.errorbar(zbelbins + 0.005, x1belbins, yerr=x1belbinserr,
             color='blue', marker='s', alpha=0.7, label='sim<10', fmt='o')
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis)
plt.ylim(x1yb[0], x1yb[1])
plt.xlabel('redshift')
plt.ylabel('x1')
plt.title('Stretch vs Redshift')
plt.legend(loc=1)
plt.text(0.02, x1yb[1] * 0.9, 'Sim Offset Mean: ' + str(decimal.Decimal(x1goffist) + decimal.Decimal(0)),
         fontsize=16)
plt.savefig('z_x14.png')
plt.show()

# random plots
plt.scatter(hostmass, c, alpha=0.5)
plt.scatter(datahostmass, datac, alpha=0.6, marker='v', color='red')
plt.xlabel('hostmass')
plt.ylabel('c')
plt.show()

plt.scatter(hostmass, x1, alpha=0.5)
plt.scatter(datahostmass, datax1, alpha=0.6, marker='v', color='red')
plt.xlabel('hostmass')
plt.ylabel('x1')
plt.show()

# heat map params
x1bounds = [7., 13., -4., 3.]
cbounds = [7., 13., -0.3, 0.3]
massrange = np.arange(x1bounds[0], x1bounds[1], 1.)
cmassrange = np.arange(x1bounds[0], x1bounds[1], 0.5)
x1range = np.arange(x1bounds[2], x1bounds[3], 1.)
crange = np.arange(cbounds[2], cbounds[3], 0.05)
crange[np.abs(crange) < 1e-10] = 0.

# heat maps
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

# plot simulated mass step
plt.scatter(hostmass, mures, alpha=0.5)
neg = np.empty(100)
pos = np.empty(100)
# these are used to plot lines representing input mass step from config.
neg.fill(-float(massstep))
pos.fill(float(massstep))
# bin mures with respect to mass
muresbel, massbuttbel = avgbins(hostmassbel, muresbel, 0.5)
muresabv, massbuttabv = avgbins(hostmassabv, muresabv, 0.5)
# plot simulated and output mass step results.
plt.scatter(massbuttabv, muresabv, color='green', s=50, marker='D', label='mass>10')
plt.scatter(massbuttbel, muresbel, color='red', s=50, marker='D', label='mass<10')
plt.plot(np.linspace(8, 12, 100), neg, color='green')
plt.plot(np.linspace(8, 12, 100), pos, color='red')
plt.xlim(8, 12)
plt.legend()
plt.xlabel('Host logmass')
plt.ylabel('mures')
plt.title('Host logmass vs mures')
plt.savefig('massstep.png')
plt.show()
