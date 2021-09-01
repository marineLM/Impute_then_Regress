"""
A figure to illustrate the discontinuities in the corrected imputed.
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                       figsize=(9, 6))

# Plot the surface of the Bayes predictor
x_grid1 = np.arange(-4.1, 4.1, 0.1)
x_grid2 = np.arange(-4.1, 4.1, 0.1)
X_grid1, X_grid2 = np.meshgrid(x_grid1, x_grid2)

Y_grid = X_grid2 ** 3 - 3 * X_grid2

surf = ax.plot_surface(X_grid1, X_grid2, Y_grid, cmap=cm.viridis,
                       linewidth=0, alpha=.5)


# Draw data point
sigma = 1.5
np.random.seed(0)
x1 = 2 * np.random.normal(size=600)
eps = np.random.normal(size=len(x1))
x2 = x1 + sigma * eps
mask = np.logical_and(np.abs(x1) < 4.1,
                      np.abs(x2) < 4.1)
x1 = x1[mask]
x2 = x2[mask]

ax.scatter(x1, x2, (Y_grid.min() - 5) * np.ones_like(x1),
           marker='o', s=10, c="C1", alpha=.3)

ax.scatter(x1, x2, x2**3 - 3 * x2,
           marker='o', s=10, c="black", alpha=.3)

ax.scatter(x1, 5.5 * np.ones_like(x2), x2**3 - 3 * x2,
           marker='o', s=20, c="C0", alpha=.4)

# Draw the Bayes predictor
x_line = np.linspace(-3.5, 3.5, 100)
ax.plot(x_line, 5.5 * np.ones_like(x_line),
        x_line ** 3 + 3 * (sigma ** 2 - 1) * x_line,
        color='k', linewidth=2, zorder=100)#label='parametric curve')

# Draw the chained oracle
x_line = np.linspace(-3.5, 3.5, 100)
ax.plot(x_line, 5.5 * np.ones_like(x_line),
        x_line ** 3 - 3 * x_line,
        color='r', linewidth=2, zorder=100)#label='parametric curve')


# Customize the axis.
ax.set_zlim(Y_grid.min() - 5, Y_grid.max())
# A StrMethodFormatter is used automatically
ax.xaxis.set_major_formatter('')
ax.yaxis.set_major_formatter('')
ax.zaxis.set_major_formatter('')
ax.view_init(elev=10., azim=-15)

ax.set_xlabel('$X_1$', labelpad=-10, size=15)
ax.set_ylabel('$X_2$', labelpad=-10, size=15)
ax.set_zlabel('Y', labelpad=-14, size=15)

# Add a label
ax.text2D(1, .76, r"$\mathbb{E}[Y|X_1]$",
          transform=ax.transAxes, ha="right",
          size=15)

ax.text2D(.92, .46, r"$f^\star\!\!\!\circ \Phi^{CI}$",
          transform=ax.transAxes, ha="right",
          size=15, color='r')


plt.subplots_adjust()
plt.savefig('../figures/corrected_imputation.png', bbox_inches='tight', dpi=200)

plt.show()



###############################################################################

fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                       figsize=(9, 6))

# Plot the surface of the Bayes predictor
x_grid1 = np.arange(-2.6, 2.6, 0.1)
x_grid2 = np.arange(-2.6, 2.6, 0.1)
X_grid1, X_grid2 = np.meshgrid(x_grid1, x_grid2)

Y_grid = X_grid2 ** 2 + X_grid1**2#

surf = ax.plot_surface(X_grid1, X_grid2, Y_grid, cmap=cm.viridis,
                       linewidth=0, alpha=.5)


# Draw data point
rho = 0.5
np.random.seed(0)
x = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=600)
x1 = x[:, 0]
x2 = x[:, 1]
mask = np.logical_and(np.abs(x1) < 2.6,
                      np.abs(x2) < 2.6)
x1 = x1[mask]
x2 = x2[mask]

ax.scatter(x1, x2, (Y_grid.min() - 2) * np.ones_like(x1),
           marker='o', s=10, c="C1", alpha=.3)

ax.scatter(x1, x2, x2**2 + x1**2,
           marker='o', s=10, c="black", alpha=.3)

ax.scatter(x1, 3.5 * np.ones_like(x2), x2**2 + x1**2,
           marker='o', s=20, c="C0", alpha=.4)

# Draw the Bayes predictor
x_line = np.linspace(-2.3, 2.3, 100)
ax.plot(x_line, 3.5 * np.ones_like(x_line),
        (1+rho**2)*(x_line ** 2 + 1),
        color='k', linewidth=2, zorder=100)#label='parametric curve')

# Draw the chained oracle
ax.plot(x_line, 3.5 * np.ones_like(x_line),
        x_line ** 2,
        color='r', linewidth=2, zorder=100)#label='parametric curve')

# Draw the imputation function
# x_line = np.linspace(-2.3, 2.3, 100)
# ax.plot(x_line, np.sqrt(rho**2*x_line**2 + (1-rho**2)),
#         (1+rho**2)*(x_line ** 2 + 1),
#         'k-.', linewidth=2, zorder=100)#label='parametric curve')

x_line = np.linspace(-2.3, 2.3, 100)
ax.plot(x_line, np.sqrt(rho**2*x_line**2 + (1-rho**2)),
        (Y_grid.min() - 2) * np.ones_like(x_line),
        'k-.', linewidth=2, zorder=100)#label='parametric curve')


# Customize the axis.
ax.set_zlim(Y_grid.min() - 2, Y_grid.max())
# A StrMethodFormatter is used automatically
ax.xaxis.set_major_formatter('')
ax.yaxis.set_major_formatter('')
ax.zaxis.set_major_formatter('')
ax.view_init(elev=10., azim=-15)

ax.set_xlabel('$X_1$', labelpad=-10, size=15)
ax.set_ylabel('$X_2$', labelpad=-10, size=15)
ax.set_zlabel('Y', labelpad=-14, size=15)

# Add a label
ax.text2D(1, .56, r"$\mathbb{E}[Y|X_1]$",
          transform=ax.transAxes, ha="right",
          size=15)

ax.text2D(.95, .32, r"$f^\star\!\!\!\circ \Phi^{CI}$",
          transform=ax.transAxes, ha="right",
          size=15, color='r')

ax.text2D(.92, .24, r"$\Phi$ corrected",
          transform=ax.transAxes, ha="right",
          size=15, color='k')


plt.subplots_adjust()
plt.savefig('../figures/corrected_imputation_bowl.png', bbox_inches='tight', dpi=200)

plt.show()

