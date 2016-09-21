import matplotlib.pyplot as plt
import scipy.interpolate as interp
import numpy as np
import decimal as d

d.getcontext().prec = 3

thein = [0.0, 0.04, 0.08, 0.16, 0.50]
theout = [0.016, 0.024, 0.064, 0.11, 0.35]
thecout = [0.011, 0.0053, 0.011, 0.0097, 0.0093]
thex1out = [-0.54, -0.51, -0.53, -0.55, -0.50]

them, theb = np.polyfit(thein, theout, 1)
trueoffset = (0.08 - theb) / them
offdec = d.Decimal((0.08 - theb) / them) + d.Decimal(0)
print('slope:', them, '  y-intercept:', theb)
print('True offset value:', trueoffset)

# plt.plot(thein, theout, color='red')
plt.plot(np.linspace(-0.1, 0.6, 5), np.linspace(-0.1, 0.6, 5) * them + theb, '--', color='black', label='Best Fit')
plt.scatter(thein, theout, color='blue', s=60, marker='D', alpha=0.5, label='Sim')
plt.scatter(trueoffset, trueoffset * them + theb, color='green', marker='v', s=80, alpha=0.6, label='8%')
plt.annotate(str(offdec), xy=(trueoffset, trueoffset * them + theb),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.9),
             xycoords='data', xytext=(trueoffset, trueoffset * them + theb + 0.1))
plt.xlabel('Input Mass Step')
plt.ylabel('Output Mass Step')
plt.xlim(-0.01, 0.6)
plt.ylim(-0.01, 0.6)
plt.legend(loc=2)
plt.show()

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
