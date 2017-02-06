import numpy as np
import os
from config import *

magfile = homedir + '/dragan/PS1Phot_absmag_bright.HOSTLIB'
massfile = homedir + '/dragan/ps1phot_mass.txt'


varname = ''
varcount = ''
with open(magfile, 'r') as f:
    rowcounter = 0
    for line in f:
        rowcounter += 1
        if 'VARNAME' in line:
            varname = line
            break
with open(magfile, 'r') as f:
    for line in f:
        if 'NVAR' in line:
            varcount = int(line.split(' ')[1])
            break
with open(magfile, 'r') as f:
    rcount = 0
    for line in f:
        rcount += 1
        if 'GAL:' in line:
            break
magdata = np.loadtxt(magfile, dtype=bytes, skiprows=rcount).astype(str)

newfile = open('ps1phot_magmass.txt', 'w')
newfile.write('GID ZTRUE LOGMASS LOGMASS_ERR g_obs r_obs i_obs z_obs g_abs r_abs i_abs z_abs\n')

with open(massfile, 'r') as f:
    next(f)
    for mass in f:
        massgid, mass, masserr = mass.split(' ')[0], mass.split(' ')[1], mass.split(' ')[2]
        for n in range(len(magdata[:, 0])):
            maggid = str(np.copy(magdata[n, 1]))
            while len(maggid) < 6:
                maggid = '0' + maggid
            if int(maggid) == int(massgid.split('c')[1]):
                ztrue = str(np.copy(magdata[n, 2]))
                mags = ''
                for i in range(10, 17, 1):
                    mags = mags + str(np.copy(magdata[n, i])) + ' '
                mags = mags + str(np.copy(magdata[n, 17]))
                newfile.write(maggid + ' ' + ztrue + ' ' + mass + ' ' + masserr.split('\n')[0] + ' ' + mags + '\n')
newfile.close()
