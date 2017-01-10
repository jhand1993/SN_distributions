import matplotlib.pyplot as plt
import scipy.interpolate as interp
import numpy as np
import decimal as d
import os
from config import *

d.getcontext().prec = 3

os.chdir(homedir + 'fitres/1KMS0_composite/')
fitres0 = '1KMS0_composite_mures+6.fitres'
with open(fitres0, 'r') as f:
    for line in f:
        if 'VARNAME' in line:
            var = line
            break
with open(fitres0, 'r') as f:
    for line in f:
        if 'NVAR' in line:
            varscount = int(line.split(' ')[1]) - 1
            break
var = var.split()
var.remove('VARNAMES:')
print(len(var))
# var.remove('FIELD')
columns = tuple(list(range(1, 5)) + list(range(6, 47)))
datasim0 = np.loadtxt(fitres0, dtype=float, skiprows=20, usecols=columns)
for i in range(varscount):
    if var[i] == 'HOST_LOGMASS':
        sim0mass = np.copy(datasim0[:, i+1])
    elif var[i] == 'x1':
        sim0x1 = np.copy(datasim0[:, i+1])
    elif var[i] == 'c':
        sim0c = np.copy(datasim0[:, i+1])
    elif var[i] == 'zCMB':
        sim0z = np.copy(datasim0[:, i+1])
    elif var[i] == 'MU':
        sim0mu = np.copy(datasim0[:, i+1])
    elif var[i] == 'MUMODEL':
        sim0mumodel = np.copy(datasim0[:, i+1])
    elif var[i] == 'MUMODEL':
        sim0mumodel = np.copy(datasim0[:, i+1])

os.chdir(homedir + 'fitres/1KMS8_composite/')
fitres8 = '1KMS8_composite_mures+6.fitres'
with open(fitres8, 'r') as f:
    for line in f:
        if 'VARNAME' in line:
            var = line
            break
with open(fitres8, 'r') as f:
    for line in f:
        if 'NVAR' in line:
            varscount = int(line.split(' ')[1]) - 2
            break
var = var.split()
var.remove('VARNAMES:')
# var.remove('FIELD')
columns = tuple(list(range(1, 5)) + list(range(6, 47)))
datasim8 = np.loadtxt(fitres8, dtype=float, skiprows=20, usecols=columns)
for i in range(varscount):
    if var[i] == 'HOST_LOGMASS':
        sim8mass = np.copy(datasim8[:, i+1])
    elif var[i] == 'x1':
        sim8x1 = np.copy(datasim8[:, i+1])
    elif var[i] == 'c':
        sim8c = np.copy(datasim8[:, i+1])
    elif var[i] == 'zCMB':
        sim8z = np.copy(datasim8[:, i+1])
    elif var[i] == 'MU':
        sim8mu = np.copy(datasim8[:, i+1])
    elif var[i] == 'MUMODEL':
        sim8mumodel = np.copy(datasim8[:, i+1])

thein = -0.01 * np.arange(0,18,2)
theout = np.array([-0.0057, -0.0323, -0.046, -0.0648, -0.0863, -0.1028, -0.1198, -0.1407, -0.1544])
thein100 = [0, -0.08]
theout100 = [-0.007, -0.068]
# thecout = [0.013, 0.010, 0.014, 0.013, 0.0066]
# thex1out = [-0.65, -0.64, -0.65, -0.62, -0.60]

them, theb = np.polyfit(thein, theout, 1)
trueoffset = (-0.08 - theb) / them
offdec = d.Decimal((-0.08 - theb) / them) + d.Decimal(0)
print('slope:', them, '  y-intercept:', theb)
print('True offset value:', trueoffset)

del0 = np.abs(np.subtract(sim0mu, sim0mumodel))
del8 = np.abs(np.subtract(sim8mu, sim8mumodel))

fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 16.0
fig_size[1] = 8.0

plt.scatter(sim0z, del0, color='blue', label='Step=0.0', alpha=0.5, marker='x')
plt.scatter(sim8z, del8, color='red', label='Step=0.08', alpha=0.5)
plt.legend(loc=2, fontsize=24)
plt.xlabel('z', fontsize=28)
plt.ylabel('mu - mumodel', fontsize=28)
plt.show()

plt.plot(np.linspace(-0.2, 0.05, 5), np.linspace(-0.2, 0.05, 5) * them + theb, '--', color='black', label='Best Fit')
plt.plot(np.linspace(-0.2, 0.05, 5), np.linspace(-0.2, 0.05, 5), '-', color='red', label='x=y', alpha=0.4)
plt.scatter(thein, theout, color='blue', s=150, marker='D', alpha=0.5, label='SNANA')
# plt.scatter(thein100, theout100, color='red', s=60, marker='D', alpha=0.8, label='Sim size=1E5')
# plt.scatter(trueoffset, trueoffset * them + theb, color='green', marker='v', s=200, alpha=0.6, label='8% output')
# plt.annotate(str(offdec), xy=(trueoffset, trueoffset * them + theb),
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.9),
#              xycoords='data', xytext=(trueoffset, trueoffset * them + theb + 0.05), fontsize=24)
plt.xlabel('Intrinsic Mass Step', fontsize=32)
plt.ylabel('Output Mass Step', fontsize=32)
plt.xlim(-0.18, 0.02)
plt.ylim(-0.18, 0.02)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc=2, fontsize=28)
plt.title('Output Mass Step vs Intrinsic Mass Step', fontsize=36)
plt.show()

"""
plt.plot(thein, thecout, color='red')
plt.scatter(thein, thecout, color='blue', s=60, marker='D')
plt.xlabel('Input Mass Step')
plt.ylabel('Output Color Step')
plt.xlim(-0.01, 0.6)
plt.ylim(-0.1, 0.1)
plt.show()

plt.plot(thein, thex1out, color='red')
plt.scatter(thein, thex1out, color='blue', s=60, marker='D')
plt.xlabel('Input Mass Step')
plt.ylabel('Output Stretch Step')
plt.xlim(-0.01, 0.6)
plt.ylim(-1, 0)
plt.show()
"""
