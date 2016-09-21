import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

# configuration parameters for scripts
homedir = '/Users/jaredhand/WFIRST_research/SN_distributions/dev'

# Dragan data
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
        datacidint = np.copy(dragan[:, i])
    elif datavars[i] == 'zCMB':
        datazcmb = np.copy(dragan[:, i])
    elif datavars[i] == 'HOST_LOGMASS':
        datahostmass = np.copy(dragan[:, i])
    elif datavars[i] == 'c':
        datac = np.copy(dragan[:, i])
    elif datavars[i] == 'x1':
        datax1 = np.copy(dragan[:, i])

# remove host_logmass < 7 from dragan data.
bad_logmass = np.asarray([x >= 7. for x in datahostmass])
datahostmass = datahostmass[bad_logmass]
datazcmb = datazcmb[bad_logmass]
datax1 = datax1[bad_logmass]
datac = datac[bad_logmass]

# import host libs
os.chdir(homedir + '/hostlib/sdss')
sdss = 'JSH_WFIRST_SDSS.HOSTLIB'
sdsslib = np.loadtxt(sdss, dtype=float, skiprows=1200, usecols=tuple([5]))
os.chdir(homedir + '/hostlib/snls')
snls = 'JSH_WFIRST_SNLS.HOSTLIB'
snlslib = np.loadtxt(snls, dtype=float, skiprows=1200, usecols=tuple([5]))
os.chdir(homedir + '/hostlib/lowz')
lowz = 'JSH_WFIRST_LOWZ.HOSTLIB'
lowzlib = np.loadtxt(lowz, dtype=float, skiprows=1200, usecols=tuple([5]))

libhostmass = np.concatenate((sdsslib, snlslib, lowzlib))

# import sim data
fitres = 'composite.fitres'
os.chdir(homedir + '/fitres/composite')
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
    elif vars[i] == 'mB':
        mb = np.copy(data[:, i])
    elif vars[i] == 'mBERR':
        mberr = np.copy(data[:, i])
print(np.mean(libhostmass), np.mean(hostmass))
print(np.mean(mberr))

libabv = np.asarray([x > 10 for x in libhostmass])
abv = np.asarray([x > 10 for x in hostmass])

print(len(libhostmass[libabv]) / len(libhostmass), len(hostmass[abv]) / len(hostmass))

figsize = plt.rcParams['figure.figsize']
figsize[0] = 10
figsize[1] = 6

plt.hist(hostmass, 100, color='blue', alpha=0.5, label='SNANA out', normed=True)
plt.hist(libhostmass, 100, color='red', alpha=0.5, label='hostlib', normed=True)
plt.legend()
plt.show()

plt.scatter(hostmass, mures, alpha=0.4)
plt.xlim(8, 12)
plt.xlabel('logmass')
plt.ylabel('mures')
plt.title('mures vs host logmass')
plt.show()
