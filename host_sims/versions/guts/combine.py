import numpy as np
import os
from config import *

# sample size
sample = 10000

# import dragan data
datafitres = 'dragan2.fitres'
os.chdir(homedir + '/dragan')
with open(datafitres, 'r') as f:
    vars = f.readlines()[1]
with open(datafitres, 'r') as f:
    varscount = int(f.readlines()[0].split(' ')[1]) - 3
vars = vars.split()
vars.remove('VARNAMES:')
vars.remove('FIELD')
vars.remove('CID')
columns = tuple(list(range(2, 5)) + list(range(6, 40)))
dragan = np.loadtxt(datafitres, dtype=float, skiprows=18, usecols=columns)
for i in range(varscount):
    if vars[i] == 'IDSURVEY':
        ids = np.copy(dragan[:, i])
    elif vars[i] == 'zCMB':
        zcmb = np.copy(dragan[:, i])
    elif vars[i] == 'HOST_LOGMASS':
        hostmass = np.copy(dragan[:, i])
    elif vars[i] == 'c':
        c = np.copy(dragan[:, i])
    elif vars[i] == 'x1':
        x1 = np.copy(dragan[:, i])
    elif vars[i] == 'MURES':
        mures = np.copy(dragan[:, i])

# remove host_logmass < 7 from dragan data.
bad_logmass = np.asarray([x >= 7. for x in hostmass])
hostmass = hostmass[bad_logmass]
zcmb = zcmb[bad_logmass]
x1 = x1[bad_logmass]
c = c[bad_logmass]
mures = mures[bad_logmass]
cidint = ids[bad_logmass]

# remove mures > abs(10) from dragan data
bad_mures = np.asarray([x < 10 for x in np.abs(mures)])
hostmass = hostmass[bad_mures]
zcmb = zcmb[bad_mures]
x1 = x1[bad_mures]
c = c[bad_mures]
mures = mures[bad_mures]
cidint = cidint[bad_mures]

# count up IDSURVEY totals for IDSURVEY=1, 4, or 15
sdss = 0
snls = 0
# ps1 = 0
for i in ids:
    if int(i) == 1:
        sdss += 1
    elif int(i) == 4:
        snls += 1
    # elif int(i) == 15:
    #     ps1 += 1

# normalize count values
sdss /= len(ids)
snls /= len(ids)
# ps1 /= len(ids)

# count up zcmb < 0.1 for all IDSURVEY != 1,4,15
lowz = 0
for i in range(len(zcmb)):
    if zcmb[i] <= 0.1 and (ids[i] != 1 or ids[i] != 4 or ids[i] != 15):
        lowz += 1

# normalize count value
lowz /= len(zcmb)

# import four sim fitres data
os.chdir(homedir + '/fitres/sdss')
sdssfitres = 'JSH_WFIRST_SDSS4_mures.fitres'
with open(sdssfitres, 'r') as f:
    sdssvars = f.readlines()[1]
with open(sdssfitres, 'r') as f:
    sdssvarscount = int(f.readlines()[0].split(' ')[1]) - 2
sdssvars = sdssvars.split()
sdssvars.remove('VARNAMES:')
sdssvars.remove('FIELD')
columns = tuple(list(range(1, 5)) + list(range(6, 56)))
sdssdata = np.loadtxt(sdssfitres, dtype=float, skiprows=12, usecols=columns)

os.chdir(homedir + '/fitres/snls')
snlsfitres = 'JSH_WFIRST_SNLS4_mures.fitres'
with open(snlsfitres, 'r') as f:
    snlsvars = f.readlines()[1]
with open(snlsfitres, 'r') as f:
    snlsvarscount = int(f.readlines()[0].split(' ')[1]) - 2
snlsvars = snlsvars.split()
snlsvars.remove('VARNAMES:')
snlsvars.remove('FIELD')
columns = tuple(list(range(1, 5)) + list(range(6, 56)))
snlsdata = np.loadtxt(snlsfitres, dtype=float, skiprows=12, usecols=columns)

"""
os.chdir(homedir + 'ps1')
ps1fitres = 'JSH_WFIRST_PS14_mures.fitres'
with open(ps1fitres, 'r') as f:
    ps1vars = f.readlines()[1]
with open(ps1fitres, 'r') as f:
    ps1varscount = int(f.readlines()[0].split(' ')[1]) - 2
ps1vars = ps1vars.split()
ps1vars.remove('VARNAMES:')
ps1vars.remove('FIELD')
columns = tuple(list(range(1, 5)) + list(range(6, 56)))
ps1data = np.loadtxt(ps1fitres, dtype=float, skiprows=12, usecols=columns)
"""
os.chdir(homedir + '/fitres/lowz')
lowzfitres = 'JSH_WFIRST_LOWZ4_mures.fitres'
with open(lowzfitres, 'r') as f:
    lowzvars = f.readlines()[1]
with open(lowzfitres, 'r') as f:
    lowzvarscount = int(f.readlines()[0].split(' ')[1]) - 2
lowzvars = lowzvars.split()
lowzvars.remove('VARNAMES:')
lowzvars.remove('FIELD')
columns = tuple(list(range(1, 5)) + list(range(6, 56)))
lowzdata = np.loadtxt(lowzfitres, dtype=float, skiprows=12, usecols=columns)

# renormalize counts
# csum = sdss + snls + ps1 + lowz
csum = sdss + snls + lowz
sdss /= csum
snls /= csum
# ps1 /= csum
lowz /= csum

# create new composite fitres with entry numbers proportioned by CID counts in dragan2.
os.chdir(homedir + '/fitres/composite')
newfitres = open('composite.fitres', mode='w')
newfitres.write('NVAR: ' + str(sdssvarscount) + '\n')
newfitres.write('VARNAMES: ' + ' '.join(sdssvars))
newfitres.write('\n')
count = 0
for i in range(len(sdssdata[:, 0])):
    if count < int(sdss * sample):
        line = 'SN: ' + ' '.join(sdssdata[i, :].astype(str)) + '\n'
        newfitres.write(line)
        count += 1
    else:
        break
count = 0
for i in range(len(snlsdata[:, 0])):
    if count < int(snls * sample):
        line = 'SN: ' + ' '.join(snlsdata[i, :].astype(str)) + '\n'
        newfitres.write(line)
        count += 1
    else:
        break
"""
count = 0
for i in range(len(ps1data[:, 0])):
    if count < int(ps1 * sample):
        line = 'SN: ' + ' '.join(ps1data[i, :].astype(str)) + '\n'
        newfitres.write(line)
        count += 1
    else:
        break
"""
count = 0
for i in range(len(lowzdata[:, 0])):
    if count < int(lowz * sample):
        line = 'SN: ' + ' '.join(lowzdata[i, :].astype(str)) + '\n'
        newfitres.write(line)
        count += 1
    else:
        break
newfitres.close()





