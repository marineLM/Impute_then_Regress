import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi
from scipy.stats import norm
from matplotlib.ticker import FormatStrFormatter

xx = np.linspace(-1, 3, 100)
y_bowl = [(x_i - 1)**2 for x_i in xx]


def f_wave(x):
    y = x - 1
    for a, b in zip([2, -4, 2], [-0.8, -1, -1.2]):
        tmp = sqrt(pi/8)*20*(x + b)
        y += a*norm.cdf(tmp)
    return y


y_wave = [f_wave(x_i) for x_i in xx]
y_disc = np.array([x_i + (x_i > 1)*3 for x_i in xx])

for y, name in [(y_bowl, 'bowl'),
                (y_wave, 'wave'),
                (y_disc, 'break')]:
    fig, ax = plt.subplots(figsize=(1.7, 1.3))
    if name != 'break':
        plt.plot(xx, y)
    else:
        plt.plot(xx[xx < 1], y[xx < 1], c='C0')
        plt.plot(xx[xx >= 1], y[xx >= 1], c='C0')
    plt.xlabel(r'$\beta^\top X + \beta_0$')
    plt.xticks([-1, 0, 1, 2, 3])
    plt.ylabel(r'f*($\beta^\top X + \beta_0$)', rotation=90)
    plt.yticks([])
    # plt.ylim(-2, 3)
    # plt.legend()
    plt.text(.5, 1, name, va='top', ha='center', transform=ax.transAxes,
             size=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(f'../figures/f_star_{name}.pdf', edgecolor='none',
                facecolor='none', bbox_inches="tight", dpi=100)
    plt.close()
# plt.show()


# Now plot the function f^\star for the counter example
xx = np.linspace(-2.1, 2.1, 100)
y = [x_i**3 - 3*x_i for x_i in xx]
fig, ax = plt.subplots(figsize=(6, 2.5))
plt.plot(xx, y)
plt.xticks([-int(2), -sqrt(3), -1, 0, 1, sqrt(3), 2])
plt.xlim(min(xx), max(xx))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
plt.yticks([-2, 0, 2])
plt.ylim(min(y), max(y))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.stem([-1, 1], [2, -2], 'k:', bottom=-3)
# plt.stem([2, -2], [-1, 1], 'k:', bottom=-3, orientation='horizontal')
xpts = [-1, 1]
ypts = [2, -2]
plt.vlines(xpts, -3, ypts, linestyle='dotted', colors='k')
plt.hlines(ypts, -3, xpts, linestyle='dotted', colors='k')
plt.scatter(xpts, ypts, zorder=2, color='k')
xpts = [-sqrt(3), 0, sqrt(3)]
ypts = [0, 0, 0]
plt.vlines(xpts, -3, ypts, linestyle='dotted', colors='orange')
plt.hlines(ypts, -3, xpts, linestyle='dotted', colors='orange')
plt.scatter(xpts, ypts, zorder=2, color='orange')
plt.savefig('../figures/f_star_cc.pdf', edgecolor='none',
            facecolor='none', bbox_inches="tight", dpi=100)
plt.close()
# plt.show()
