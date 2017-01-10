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

# import sim1 data
fitres0 = '100KMS0_composite.fitres'
os.chdir(homedir + '/fitres/100KMS0_composite')
with open(fitres0, 'r') as f:
    for line in f:
        if 'VARNAME' in line:
            var = line
            break
with open(fitres0, 'r') as f:
    for line in f:
        if 'NVAR' in line:
            varscount = int(line.split(' ')[1])
            break
var = var.split()
var.remove('VARNAMES:')
# vars.remove('FIELD')
print(var)
print(len(var), varscount)
columns = tuple(list(range(1, 51)))
rowskipnum = 2
data0 = np.loadtxt(fitres0, dtype=float, skiprows=rowskipnum, usecols=columns)
for i in range(varscount):
    if var[i] == 'MURES':
        mures0 = np.copy(data0[:, i])
    elif var[i] == 'HOST_LOGMASS':
        hostmass0 = np.copy(data0[:, i])
    elif var[i] == 'zCMB':
        zcmb0 = np.copy(data0[:, i])
    elif var[i] == 'x1':
        x10 = np.copy(data0[:, i])
    elif var[i] == 'c':
        c0 = np.copy(data0[:, i])
    elif var[i] == 'FITPROB':
        fitprob0 = np.copy(data0[:, i])
# print(max(mures), min(mures))
# print(max(c), min(c))
# print(max(x1), min(x1))
bad_c0 = np.asarray([x < 0.3 for x in np.abs(c0)])
c0 = c0[bad_c0]
x10 = x10[bad_c0]
zcmb0 = zcmb0[bad_c0]
mures0 = mures0[bad_c0]
hostmass0 = hostmass0[bad_c0]
fitprob0 = fitprob0[bad_c0]

bad_x10 = np.asarray([x < 4. for x in np.abs(x10)])
c0 = c0[bad_x10]
x10 = x10[bad_x10]
zcmb0 = zcmb0[bad_x10]
mures0 = mures0[bad_x10]
hostmass0 = hostmass0[bad_x10]
fitprob = fitprob0[bad_x10]

bad_mures0 = np.asarray([x < 1. for x in np.abs(mures0)])
c0 = c0[bad_mures0]
x10 = x10[bad_mures0]
zcmb0 = zcmb0[bad_mures0]
mures0 = mures0[bad_mures0]
hostmass0 = hostmass0[bad_mures0]
fitprob0 = fitprob0[bad_mures0]

bad_fitprob0 = np.asarray([x > 0.01 for x in np.abs(fitprob0)])
c0 = c0[bad_fitprob0]
x10 = x10[bad_fitprob0]
zcmb0 = zcmb0[bad_fitprob0]
mures0 = mures0[bad_fitprob0]
hostmass0 = hostmass0[bad_fitprob0]
fitprob0 = fitprob0[bad_fitprob0]

# import sim2 data
fitres8 = '100KMS8_composite.fitres'
os.chdir(homedir + '/fitres/100KMS8_composite')
with open(fitres8, 'r') as f:
    for line in f:
        if 'VARNAME' in line:
            var = line
            break
with open(fitres8, 'r') as f:
    for line in f:
        if 'NVAR' in line:
            varscount = int(line.split(' ')[1])
            break
var = var.split()
var.remove('VARNAMES:')
# vars.remove('FIELD')
data8 = np.loadtxt(fitres8, dtype=float, skiprows=rowskipnum, usecols=columns)
for i in range(varscount):
    if var[i] == 'MURES':
        mures8 = np.copy(data8[:, i])
    elif var[i] == 'HOST_LOGMASS':
        hostmass8 = np.copy(data8[:, i])
    elif var[i] == 'zCMB':
        zcmb8 = np.copy(data8[:, i])
    elif var[i] == 'x1':
        x18 = np.copy(data8[:, i])
    elif var[i] == 'c':
        c8 = np.copy(data8[:, i])
    elif var[i] == 'FITPROB':
        fitprob8 = np.copy(data8[:, i])
# print(max(mures), min(mures))
# print(max(c), min(c))
# print(max(x1), min(x1))
bad_c8 = np.asarray([x < 0.3 for x in np.abs(c8)])
c8 = c8[bad_c8]
x18 = x18[bad_c8]
zcmb8 = zcmb8[bad_c8]
mures8 = mures8[bad_c8]
hostmass8 = hostmass8[bad_c8]
fitprob8 = fitprob8[bad_c8]

bad_x18 = np.asarray([x < 4. for x in np.abs(x18)])
c8 = c8[bad_x18]
x18 = x18[bad_x18]
zcmb8 = zcmb8[bad_x18]
mures8 = mures8[bad_x18]
hostmass8 = hostmass8[bad_x18]
fitprob = fitprob8[bad_x18]

bad_mures8 = np.asarray([x < 1. for x in np.abs(mures8)])
c8 = c8[bad_mures8]
x18 = x18[bad_mures8]
zcmb8 = zcmb8[bad_mures8]
mures8 = mures8[bad_mures8]
hostmass8 = hostmass8[bad_mures8]
fitprob8 = fitprob8[bad_mures8]

bad_fitprob8 = np.asarray([x > 0.01 for x in np.abs(fitprob8)])
c8 = c8[bad_fitprob8]
x18 = x18[bad_fitprob8]
zcmb8 = zcmb8[bad_fitprob8]
mures8 = mures8[bad_fitprob8]
hostmass8 = hostmass8[bad_fitprob8]
fitprob8 = fitprob8[bad_fitprob8]

# find mures above and below hostmass=18:
aboveten0 = np.asarray([x >= 10 for x in hostmass0])
belowten0 = np.asarray([x < 10 for x in hostmass0])

aboveten8 = np.asarray([x >= 10 for x in hostmass8])
belowten8 = np.asarray([x < 10 for x in hostmass8])

"""
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
"""
# find mures above and below hostmass=10 for dragan data
dataaboveten = np.asarray([x >= 10 for x in datahostmass])
databelowten = np.asarray([x < 10 for x in datahostmass])

"""
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
"""
# split up sim mures, x1, c and zcmb based on mass greater or less than 10
muresabv0 = np.asarray(mures0[aboveten0])
muresbel0 = np.asarray(mures0[belowten0])
cabv0 = np.asarray(c0[aboveten0])
cbel0 = np.asarray(c0[belowten0])
x1abv0 = np.asarray(x10[aboveten0])
x1bel0 = np.asarray(x10[belowten0])
zabv0 = np.asarray(zcmb0[aboveten0])
zbel0 = np.asarray(zcmb0[belowten0])
hostmassabv0 = np.asarray(hostmass0[aboveten0])
hostmassbel0 = np.asarray(hostmass0[belowten0])

muresabv8 = np.asarray(mures8[aboveten8])
muresbel8 = np.asarray(mures8[belowten8])
cabv8 = np.asarray(c8[aboveten8])
cbel8 = np.asarray(c8[belowten8])
x1abv8 = np.asarray(x18[aboveten8])
x1bel8 = np.asarray(x18[belowten8])
zabv8 = np.asarray(zcmb8[aboveten8])
zbel8 = np.asarray(zcmb8[belowten8])
hostmassabv8 = np.asarray(hostmass8[aboveten8])
hostmassbel8 = np.asarray(hostmass8[belowten8])

print('MS0:', np.mean(muresabv0)-np.mean(muresbel0))
print('MS8:', np.mean(muresabv8)-np.mean(muresbel8))

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
"""
# make cells for heat map
massx1bounds = [7., 13., -4., 3.]
masscbounds = [7., 13., -0.3, 0.3]
x1massmap = digitize(hostmass, x1, massx1bounds)
cmassmap = digitize(hostmass, c, masscbounds, step1=0.5, step2=0.05)
datax1massmap = digitize(datahostmass, datax1, massx1bounds)
datacmassmap = digitize(datahostmass, datac, masscbounds, step1=0.5, step2=0.05)
"""
# zbin size
z = 0.1
# bin mures with respect to mass
muresbins20, massbins20, massbinserr20 = avgbins(hostmass0, mures0, 1, sem=True)
# bin mures with respect to z
muresabvbins0, zabvbins0, muresabvbinserr0 = avgbins(zabv0, muresabv0, z, sem=True)
muresbelbins0, zbelbins0, muresbelbinserr0 = avgbins(zbel0, muresbel0, z, sem=True)
muresbins0, zbins0, muresbinserr0 = avgbins(zcmb0, mures0, z, sem=True)
muresabvbins0[muresabvbins0 == 0] = np.nan
muresbelbins0[muresbelbins0 == 0] = np.nan
muresbins0[muresbins0 == 0] = np.nan
muresabvbinserr0[muresabvbinserr0 == 0] = np.nan
muresbelbinserr0[muresbelbinserr0 == 0] = np.nan
muresbinserr0[muresbinserr0 == 0] = np.nan
# bin c with respect to z
cabvbins0, zabvbins0, cabvbinserr0 = avgbins(zabv0, cabv0, z, sem=True)
cbelbins0, zbelbins0, cbelbinserr0 = avgbins(zbel0, cbel0, z, sem=True)
cbins0, zbins0, cbinserr0 = avgbins(zcmb0, c0, z, sem=True)
cabvbins0[cabvbins0 == 0] = np.nan
cbelbins0[cbelbins0 == 0] = np.nan
cbins0[cbins0 == 0] = np.nan
cabvbinserr0[cabvbinserr0 == 0] = np.nan
cbelbinserr0[cbelbinserr0 == 0] = np.nan
cbinserr0[cbinserr0 == 0] = np.nan
# bin x1 with respect to z
x1abvbins0, zabvbins0, x1abvbinserr0 = avgbins(zabv0, x1abv0, z, sem=True)
x1belbins0, zbelbins0, x1belbinserr0 = avgbins(zbel0, x1bel0, z, sem=True)
x1bins0, zbins0, x1binserr0 = avgbins(zcmb0, x10, z, sem=True)
x1abvbins0[x1abvbins0 == 0] = np.nan
x1belbins0[x1belbins0 == 0] = np.nan
x1bins0[x1bins0 == 0] = np.nan
x1abvbinserr0[x1abvbinserr0 == 0] = np.nan
x1belbinserr0[x1belbinserr0 == 0] = np.nan
x1binserr0[x1binserr0 == 0] = np.nan

# bin mures with respect to mass
muresbins28, massbins28, massbinserr28 = avgbins(hostmass8, mures8, 1, sem=True)
# bin mures with respect to z
muresabvbins8, zabvbins8, muresabvbinserr8 = avgbins(zabv8, muresabv8, z, sem=True)
muresbelbins8, zbelbins8, muresbelbinserr8 = avgbins(zbel8, muresbel8, z, sem=True)
muresbins8, zbins8, muresbinserr8 = avgbins(zcmb8, mures8, z, sem=True)
muresabvbins8[muresabvbins8 == 0] = np.nan
muresbelbins8[muresbelbins8 == 0] = np.nan
muresbins8[muresbins8 == 0] = np.nan
muresabvbinserr8[muresabvbinserr8 == 0] = np.nan
muresbelbinserr8[muresbelbinserr8 == 0] = np.nan
muresbinserr8[muresbinserr8 == 0] = np.nan
# bin c with respect to z
cabvbins8, zabvbins8, cabvbinserr8 = avgbins(zabv8, cabv8, z, sem=True)
cbelbins8, zbelbins8, cbelbinserr8 = avgbins(zbel8, cbel8, z, sem=True)
cbins8, zbins8, cbinserr8 = avgbins(zcmb8, c8, z, sem=True)
cabvbins8[cabvbins8 == 0] = np.nan
cbelbins8[cbelbins8 == 0] = np.nan
cbins8[cbins8 == 0] = np.nan
cabvbinserr8[cabvbinserr8 == 0] = np.nan
cbelbinserr8[cbelbinserr8 == 0] = np.nan
cbinserr8[cbinserr8 == 0] = np.nan
# bin x1 with respect to z
x1abvbins8, zabvbins8, x1abvbinserr8 = avgbins(zabv8, x1abv8, z, sem=True)
x1belbins8, zbelbins8, x1belbinserr8 = avgbins(zbel8, x1bel8, z, sem=True)
x1bins8, zbins8, x1binserr8 = avgbins(zcmb8, x18, z, sem=True)
x1abvbins8[x1abvbins8 == 0] = np.nan
x1belbins8[x1belbins8 == 0] = np.nan
x1bins8[x1bins8 == 0] = np.nan
x1abvbinserr8[x1abvbinserr8 == 0] = np.nan
x1belbinserr8[x1belbinserr8 == 0] = np.nan
x1binserr8[x1binserr8 == 0] = np.nan

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

"""
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


print('Sim mass distribution parameters:')
print('mbar:', mp[0], 'sigma pos:', mp[1], 'sigma neg:', mp[2])
print('Data mass distribution parameters:')
print('mbar:', datamp[0], 'sigma pos:', datamp[1], 'sigma neg:', datamp[2])
print('')

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
"""
# plots...
fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 16.0
fig_size[1] = 8.0
zaxis = np.arange(0., 1., 0.1)

# set decimal precision for plot annotations
decimal.getcontext().prec = 2
os.chdir(homedir + '/mures_plots/composite')

# y-axis bounds for plots
myb = [-0.05, 0.05]
cyb = [-0.2, 0.2]
x1yb = [-1.5, 1.5]

plt.errorbar(zbins0 + 0.005, muresbins0, yerr=muresbinserr0, markersize=12,
             color='red', marker='D', alpha=0.9, label='Mass step = 0%', fmt='o')
plt.errorbar(zbins8 - 0.005, muresbins8, yerr=muresbinserr8, markersize=12,
             color='blue', marker='D', alpha=0.9,  label='Mass step = 8%', fmt='o')
# plt.scatter(zcmb0, mures0, alpha=0.4, color='g', label='MS0', marker='x')
# plt.scatter(zcmb8, mures8, alpha=0.4, color='b', label='MS8', marker='x')
plt.plot([0,1], [0,0], '-', color='black')
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis, fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(myb[0], myb[1])
plt.xlabel('Redshift', fontsize=32)
plt.ylabel('Hubble residuals (mag)', fontsize=32)
plt.title('Does Distance Bias Depend on Mass Step?', fontsize=36)
plt.legend(loc=1, fontsize=24)
plt.show()

# plt.errorbar(zabvbins0 + 0.005, muresabvbins0, yerr=muresabvbinserr0, markersize=12,
#              color='red', marker='D', alpha=0.7, label=r'$M_H > 10$, Mass step = 0%', fmt='o')
# plt.errorbar(zbelbins0 + 0.005, muresbelbins0, yerr=muresbelbinserr0, markersize=12,
#              color='blue', marker='s', alpha=0.7, label=r'$M_H < 10$, Mass step = 0%', fmt='o')
plt.errorbar(zabvbins8 - 0.005, muresabvbins8, yerr=muresabvbinserr8, markersize=12,
             color='red', marker='D', alpha=0.7, label=r'$M_H > 10$, Mass step = 8%', fmt='o')
plt.errorbar(zbelbins8 - 0.005, muresbelbins8, yerr=muresbelbinserr8, markersize=12,
             color='blue', marker='s', alpha=0.7, label=r'$M_H < 10$, Mass step = 8%', fmt='o')
plt.plot([0,1], [0,0], '-', color='black')
plt.xlim(0., 1.)
plt.xticks(zaxis, zaxis, fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(myb[0], myb[1])
plt.xlabel('Redshift', fontsize=30)
plt.ylabel('Hubble residuals (mag)', fontsize=32)
plt.title('How does Mass Step change with Redshift?', fontsize=36)
plt.legend(loc=1, fontsize=24)
# plt.savefig('z_mures3.png')
plt.show()
