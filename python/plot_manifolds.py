"""
Part 1: A figure to illustrate the manifolds corresponding to corrected
imputations for the bowl function in 3D, with Gaussian data.

Part 2: A figure to illustrate the manifolds corresponding to imputations by
conditional expectation with Gaussian data.
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


# Part 1: corrected imputations for the bowl function
######################################################
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                       figsize=(9, 6))

x_grid1 = np.arange(-4.1, 4.1, 0.1)
x_grid2 = np.arange(-4.1, 4.1, 0.1)
x_grid3 = np.arange(-4.1, 4.1, 0.1)

rho = 0.5
q = 1/(1-rho**2)

# Draw data points
Sigma = [[1, rho, 0], [rho, 1, rho], [0, rho, 1]]
np.random.seed(0)
x = np.random.multivariate_normal(mean=[0, 0, 0], cov=Sigma, size=600)
mask = np.logical_and(np.abs(x[:, 0]) < 4.1,
                      np.abs(x[:, 1]) < 4.1,
                      np.abs(x[:, 2]) < 4.1)
x = x[mask]


# Plot 2D manifolds
X_grid1, X_grid2 = np.meshgrid(x_grid1, x_grid2)
X3_imp = np.sqrt(1 - rho**2*q + (q*(-rho**2*X_grid1 + rho*X_grid2))**2)
surf1 = ax.plot_surface(X_grid1, X_grid2, X3_imp, cmap=cm.viridis,
                        linewidth=0, alpha=.5)

X_grid2, X_grid3 = np.meshgrid(x_grid2, x_grid3)
X1_imp = np.sqrt(1 - rho**2*q + (q*(rho*X_grid2 - rho**2*X_grid3))**2)
surf2 = ax.plot_surface(X1_imp, X_grid2, X_grid3, cmap=cm.viridis,
                        linewidth=0, alpha=.5)

X_grid1, X_grid3 = np.meshgrid(x_grid1, x_grid3)
X2_imp = np.sqrt(1 - 2*rho**2 + (rho*X_grid1 + rho*X_grid3)**2)
surf2 = ax.plot_surface(X_grid1, X2_imp, X_grid3, cmap=cm.viridis,
                        linewidth=0, alpha=.5)

# Plot 1D manifold
# Only x1 observed
x2_imp = np.sqrt(1 - rho**2 + (rho*x_grid1)**2)
x3_imp = np.ones_like(x_grid1)
line1 = ax.plot(x_grid1, x2_imp, x3_imp,
                color='k', linewidth=1, zorder=100)

# Only x2 observed
x1_imp = np.sqrt(1 - rho**2 + (rho*x_grid2)**2)
x3_imp = np.sqrt(1 - rho**2 + (rho*x_grid2)**2)
line2 = ax.plot(x1_imp, x_grid2, x3_imp,
                color='k', linewidth=1, zorder=100)

# Only x3 observed
x2_imp = np.sqrt(1 - rho**2 + (rho*x_grid3)**2)
x1_imp = np.ones_like(x_grid1)
line3 = ax.plot(x1_imp, x2_imp, x_grid3,
                color='k', linewidth=1, zorder=100)

# Customize the axis.
# A StrMethodFormatter is used automatically
ax.xaxis.set_major_formatter('')
ax.yaxis.set_major_formatter('')
ax.zaxis.set_major_formatter('')
ax.view_init(elev=10., azim=-15)

ax.set_xlabel('$X_1$', labelpad=-10, size=15)
ax.set_ylabel('$X_2$', labelpad=-10, size=15)
ax.set_zlabel('$X_3$', labelpad=-10, size=15)


plt.subplots_adjust()
plt.savefig('../figures/manifolds_corrected.png', bbox_inches='tight', dpi=200)

# plt.show()
plt.close()


# Part 1: corrected imputations for the bowl function - data points version
###########################################################################
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                       figsize=(9, 6))

rho = 0.5
q = 1/(1-rho**2)

# Draw data points
Sigma = [[1, rho, 0], [rho, 1, rho], [0, rho, 1]]
np.random.seed(0)
x = np.random.multivariate_normal(mean=[0, 0, 0], cov=Sigma, size=700)
x1 = x[:, 0]
x2 = x[:, 1]
x3 = x[:, 2]

# Plot 3D manifold
# ax.scatter(x1, x2, x3, marker='o', s=10, c="k", alpha=.3)

# Plot 2D manifolds
x3_imp = np.sqrt(1 - rho**2*q + (q*(-rho**2*x1 + rho*x2))**2)
ax.scatter(x1, x2, x3_imp, marker='o', s=10, c="C0", alpha=.3)

x1_imp = np.sqrt(1 - rho**2*q + (q*(rho*x2 - rho**2*x3))**2)
ax.scatter(x1_imp, x2, x3, marker='o', s=10, c="C1", alpha=.3)


x2_imp = np.sqrt(1 - 2*rho**2 + (rho*x1 + rho*x3)**2)
ax.scatter(x1, x2_imp, x3, marker='o', s=10, c="C2", alpha=.3)

# Plot 1D manifold
# Only x1 observed
x2_imp = np.sqrt(1 - rho**2 + (rho*x1)**2)
x3_imp = np.ones_like(x1)
ax.scatter(x1, x2_imp, x3_imp, marker='o', s=10, c="C3", alpha=.3)

# Only x2 observed
x1_imp = np.sqrt(1 - rho**2 + (rho*x2)**2)
x3_imp = np.sqrt(1 - rho**2 + (rho*x2)**2)
ax.scatter(x1_imp, x2, x3_imp, marker='o', s=10, c="C4", alpha=.3)

# Only x3 observed
x2_imp = np.sqrt(1 - rho**2 + (rho*x3)**2)
x1_imp = np.ones_like(x1)
ax.scatter(x1_imp, x2_imp, x3, marker='o', s=10, c="C5", alpha=.3)

# Customize the axis.
# A StrMethodFormatter is used automatically
ax.xaxis.set_major_formatter('')
ax.yaxis.set_major_formatter('')
ax.zaxis.set_major_formatter('')
ax.view_init(elev=10., azim=-15)

ax.set_xlabel('$X_1$', labelpad=-10, size=15)
ax.set_ylabel('$X_2$', labelpad=-10, size=15)
ax.set_zlabel('$X_3$', labelpad=-10, size=15)


plt.subplots_adjust()
plt.savefig('../figures/manifolds_corrected_points.png',
            bbox_inches='tight', dpi=200)

# plt.show()
plt.close()


# Part 2: Imputations by the conditional expctation
#####################################################
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                       figsize=(9, 6))

x_grid1 = np.arange(-4.1, 4.1, 0.1)
x_grid2 = np.arange(-4.1, 4.1, 0.1)
x_grid3 = np.arange(-4.1, 4.1, 0.1)

rho = 0.5
q = rho/(1-rho**2)

# Plot 2D manifolds
X_grid1, X_grid2 = np.meshgrid(x_grid1, x_grid2)
X3_imp = q*(-rho**2*X_grid1 + rho*X_grid2)
surf1 = ax.plot_surface(X_grid1, X_grid2, X3_imp, cmap=cm.viridis,
                        linewidth=0, alpha=.5)

X_grid2, X_grid3 = np.meshgrid(x_grid2, x_grid3)
X1_imp = q*(rho*X_grid2 - rho**2*X_grid3)
surf2 = ax.plot_surface(X1_imp, X_grid2, X_grid3, cmap=cm.viridis,
                        linewidth=0, alpha=.5)

X_grid1, X_grid3 = np.meshgrid(x_grid1, x_grid3)
X2_imp = rho*X_grid1 + rho*X_grid3
surf2 = ax.plot_surface(X_grid1, X2_imp, X_grid3, cmap=cm.viridis,
                        linewidth=0, alpha=.5)

# Plot 1D manifold
# Only x1 observed
x2_imp = rho*x_grid1
x3_imp = np.zeros_like(x_grid1)
line1 = ax.plot(x_grid1, x2_imp, x3_imp,
                color='k', linewidth=1, zorder=100)

# Only x2 observed
x1_imp = rho*x_grid2
x3_imp = rho*x_grid2
line2 = ax.plot(x1_imp, x_grid2, x3_imp,
                color='k', linewidth=1, zorder=100)

# Only x3 observed
x2_imp = rho*x_grid3
x1_imp = np.zeros_like(x_grid1)
line3 = ax.plot(x1_imp, x2_imp, x_grid3,
                color='k', linewidth=1, zorder=100)

# Customize the axis.
# A StrMethodFormatter is used automatically
ax.xaxis.set_major_formatter('')
ax.yaxis.set_major_formatter('')
ax.zaxis.set_major_formatter('')
ax.view_init(elev=10., azim=-15)

ax.set_xlabel('$X_1$', labelpad=-10, size=15)
ax.set_ylabel('$X_2$', labelpad=-10, size=15)
ax.set_zlabel('$X_3$', labelpad=-10, size=15)


plt.subplots_adjust()
plt.savefig('../figures/manifolds_CI.png', bbox_inches='tight', dpi=200)

# plt.show()
plt.close()
