import itertools

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches, lines
import numpy as np


def plot_one(ax, data, methods=None, colors=None, is_legend=False, dim=False):

    if is_legend:
        is_legend = 'full'

    # sns.set_palette('bright')

    if methods is None:
        methods = data['method'].unique()

    # Plot test
    #sns.boxplot(
    #    data=data.query('train_test == "test"'), x='r2', order=methods,
    #            saturation=1, y='method', ax=ax, boxprops=dict(alpha=alpha),
    #            whiskerprops=dict(alpha=alpha),
    #            flierprops=dict(alpha=alpha))
    # sns.violinplot(
    #     data=data, x='R2_test', order=methods,
    #             saturation=1, y='method', ax=ax,
    #             scale="width",
    #             color=('.8' if dim else None))
    sns.violinplot(
        data=data, x='R2_test', order=methods, inner='points',
                saturation=1,y='method', ax=ax,
                scale="width",
                palette = colors,
                color=('.8' if dim else None))

    # for i in range(len(methods)):
    #     if i % 2:
    #         ax.axhspan(i - .5, i + .5, color='.9', zorder=0)

    # Plot train
    # sns.lineplot(
    #    data=data.query('train_test == "train"'), x='params_by_samples',
    #    y='r2', hue='experiment', ax=ax, ci=None,
    #    legend=False, style="train_test", dashes=6 * [(1, 2)])

    # Set axes
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.axvline(0, color='.1', linewidth=1)
    ax.spines['right'].set_edgecolor('.6')
    ax.spines['left'].set_edgecolor('.6')
    ax.spines['top'].set_edgecolor('.6')
    ax.set_ylim(len(methods) - .5, -.5)
    # ax.grid(True)


NAMES = {
    'BayesPredictor_order0': 'Chaining oracles',
    'oracleMLP': 'Oracle impute + MLP',
    'NeuMiss_shared_custom_normal': 'NeuMiss + MLP',
    'MICEMLP': 'MICE + MLP',
    'MICEMLP_mask': 'MICE & mask + MLP',
    'meanMLP': 'mean impute + MLP',
    'meanMLP_mask': 'mean impute & mask + MLP',
    'GBRT': 'Gradient-boosted trees',
}


if __name__ == '__main__':
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['xtick.major.pad'] = .7
    plt.rcParams['ytick.major.pad'] = 2
    plt.rcParams['ytick.major.size'] = 0
    plt.rcParams['xtick.major.size'] = 2

    # fig, axes = plt.subplots(
    #         2, 4, figsize=(10, 6), sharex='col', sharey=True)
    # fig.suptitle(data_type)

    # for data_type in ['MCAR_square', 'MCAR_cube', 'gaussian_sm_square',
    #                   'gaussian_sm_cube']:
    # for data_type in ['MCAR_square', 'MCAR_stairs', 'gaussian_sm_square',
    #                   'gaussian_sm_stairs']:
    # for data_size in ['_2500', '']:
    #    for data_type in ['MCAR_square', 'MCAR_stairs', 'gaussian_sm_square',
    #                   'gaussian_sm_stairs']:
    #     data_type += data_size

    scores = pd.read_csv(f'../results/california_MCAR.csv',
                            index_col=0)

    n_sizes = scores['n'].unique()

    fig, axes = plt.subplots(
            1, len(n_sizes), figsize=(10, 6), sharex='col', sharey=True)

    # Separate Bayes rate from other methods performances
    # br = scores.query('method == "BayesPredictor"')
    # scores = scores.query('method != "BayesPredictor"')

    # Choose the methods to be plotted
    # methods = ['BayesPredictor_order0', 'oracleMLP', 'oracleMLP_mask',
    #            'MICEMLP', 'MICEMLP_mask', 'NeuMiss_shared', 'meanMLP',
    #            'meanMLP_mask', 'GBRT']

    # Differentiate NeuMiss according to the type of initialization
    # ind_custom_normal = (scores.method == "NeuMiss_shared") & \
    #                     (scores.init_type == "custom_normal")
    # ind_custom_uniform = (scores.method == "NeuMiss_shared") & \
    #                      (scores.init_type == "uniform")
    # scores.loc[ind_custom_normal, 'method'] = 'NeuMiss_shared_custom_normal'
    # scores.loc[ind_custom_uniform, 'method'] = 'NeuMiss_shared_uniform'

    methods = ['BayesPredictor', 'BayesPredictor_order0',
               'oracleMLP', 'oracleMLPPytorch',
               'oracleMLP_mask', 'oracleMLPPytorch_mask',
               'NeuMiss_shared_custom_normal',
               'NeuMiss_shared_uniform',
               'NeuMiss_shared',
               'NeuMiss_normal_', 'NeuMiss_normal_mask',
               'NeuMiss_custom_normal_', 'NeuMiss_custom_normal_mask',
               'NeuMiss_uniform_', 'NeuMiss_uniform_mask',
               'MICEMLP','MICEMLPPytorch',
               'MICEMLP_mask', 'MICEMLPPytorch_mask',
               'meanMLP', 'meanMLPPytorch',
               'meanMLP_mask', 'meanMLPPytorch_mask',
               'GBRT']
    palette = sns.color_palette("hls", len(methods), desat=1)          
    colors = dict()
    for method, col in zip(methods, palette):
        colors[method] = col
    scores = scores.query('method in @methods')

    # # Choose one lr and one weight decay for Pytorch based methods
    # ind_lr = (scores.lr == 1e-4) | np.isnan(scores.lr)
    # scores = scores.loc[ind_lr, ]
    # ind_wd = (scores.weight_decay == 1e-4) | np.isnan(scores.weight_decay)
    # scores = scores.loc[ind_wd, ]


    # # Find the best depth for NeuMiss_shared,
    # # and the best depth for MICEMLP and meanMLP.
    # # scores_no_na = scores.query('train_test == "test"')
    # scores_no_na = scores.copy()
    # scores_no_na['depth'] = scores_no_na['depth'].fillna(value=0)
    # scores_no_na['mlp_depth'] = scores_no_na['mlp_depth'].fillna(value=0)
    # # Averaging over iterations
    # mean_score = scores_no_na.groupby(
    #     ['method', 'n', 'prop_latent', 'depth', 'mlp_depth']
    #                                     )['R2_val'].median()
    # mean_score = mean_score.reset_index()
    # mean_score = mean_score.sort_values(
    #     by=['method', 'n', 'prop_latent', 'R2_val'])
    # best_depth = mean_score.groupby(
    #     ['method', 'n', 'prop_latent']).last()[['depth', 'mlp_depth']]
    # best_depth = best_depth.rename(
    #     columns={'depth': 'best_depth', 'mlp_depth': 'best_mlp_depth'})
    # scores_no_na = scores_no_na.set_index(['method', 'n', 'prop_latent']
    #                                         ).join(best_depth)
    # scores_no_depth = scores_no_na.reset_index()
    # scores_no_depth = scores_no_depth.query(
    #     'depth == best_depth and mlp_depth == best_mlp_depth')

    
    # Find the best depth for NeuMiss_shared,
    # and the best depth for MICEMLP and meanMLP.
    # and the best lr and weight decay for Pytorch-based methods
    # scores_no_na = scores.query('train_test == "test"')
    # and the best width_factor
    scores_no_na = scores.copy()
    scores_no_na['depth'] = scores_no_na['depth'].fillna(value=0)
    scores_no_na['mlp_depth'] = scores_no_na['mlp_depth'].fillna(value=0)
    scores_no_na['lr'] = scores_no_na['lr'].fillna(value=0)
    scores_no_na['weight_decay'] = scores_no_na['weight_decay'].fillna(value=0)
    scores_no_na['width_factor'] = scores_no_na['width_factor'].fillna(value=0)
    # Averaging over iterations
    mean_score = scores_no_na.groupby(
        ['method', 'n', 'prop_latent', 'depth', 'mlp_depth', 'lr',
         'weight_decay', 'width_factor'])['R2_val'].median()
    mean_score = mean_score.reset_index()
    mean_score = mean_score.sort_values(
        by=['method', 'n', 'prop_latent', 'R2_val'])
    best_depth = mean_score.groupby(
        ['method', 'n', 'prop_latent']).last()[['depth', 'mlp_depth', 'lr', 'weight_decay', 'width_factor']]
    best_depth = best_depth.rename(
        columns={'depth': 'best_depth', 'mlp_depth': 'best_mlp_depth',
                 'lr': 'best_lr', 'weight_decay': 'best_weight_decay',
                 'width_factor': 'best_width_factor'})
    scores_no_na = scores_no_na.set_index(['method', 'n', 'prop_latent']
                                            ).join(best_depth)
    scores_no_depth = scores_no_na.reset_index()
    tmp = ('depth == best_depth and mlp_depth == best_mlp_depth' +
           ' and lr == best_lr and weight_decay == best_weight_decay' +
           ' and width_factor == best_width_factor')
    scores_no_depth = scores_no_depth.query(tmp)

    # For real data where the is no prop_latent variable
    # ------------------------------------------------
    # Find the best depth for NeuMiss_shared,
    # and the best depth for MICEMLP and meanMLP.
    # and the best lr and weight decay for Pytorch-based methods
    # scores_no_na = scores.query('train_test == "test"')
    scores_no_na = scores.copy()
    scores_no_na['depth'] = scores_no_na['depth'].fillna(value=0)
    scores_no_na['mlp_depth'] = scores_no_na['mlp_depth'].fillna(value=0)
    scores_no_na['lr'] = scores_no_na['lr'].fillna(value=0)
    scores_no_na['weight_decay'] = scores_no_na['weight_decay'].fillna(value=0)
    scores_no_na['width_factor'] = scores_no_na['width_factor'].fillna(value=0)
    # Averaging over iterations
    mean_score = scores_no_na.groupby(
        ['method', 'n', 'depth', 'mlp_depth', 'lr', 'weight_decay', 
         'width_factor'])['R2_val'].mean()
    mean_score = mean_score.reset_index()
    mean_score = mean_score.sort_values(
        by=['method', 'n', 'R2_val'])
    best_depth = mean_score.groupby(
        ['method', 'n']).last()[['depth', 'mlp_depth', 'lr', 'weight_decay', 'width_factor']]
    best_depth = best_depth.rename(
        columns={'depth': 'best_depth', 'mlp_depth': 'best_mlp_depth',
                 'lr': 'best_lr', 'weight_decay': 'best_weight_decay',
                 'width_factor': 'best_width_factor'})
    scores_no_na = scores_no_na.set_index(['method', 'n']
                                            ).join(best_depth)
    scores_no_depth = scores_no_na.reset_index()
    tmp = ('depth == best_depth and mlp_depth == best_mlp_depth' +
           ' and lr == best_lr and weight_decay == best_weight_decay' +
           ' and width_factor == best_width_factor')
    scores_no_depth = scores_no_depth.query(tmp)

    # Correct by the Bayes rate
    # br_test = br.query('train_test == "test"')
    # for it in scores_no_depth.iter.unique():
    #     for prop_latent in scores_no_depth.prop_latent.unique():
    #         br_val = float(br.query('iter == @it and prop_latent == \
    #              @prop_latent and train_test == "test"')['r2'])
    #         ind = (scores_no_depth.iter == it) & \
    #               (scores_no_depth.prop_latent == prop_latent)
    #         scores_no_depth.loc[ind, 'r2'] = scores_no_depth.loc[ind, 'r2'] - br_val

    # scores_with_br['method'] = scores_with_br['method'].replace({
    #     'EMLR': 'EM',
    #     'MICELR': 'MICE + LR',
    #     'MICEMLP': 'MICE + MLP',
    #     'torchMLP': 'MLP',
    #     })

    # if 'square' in data_type:
    #     j = 0
    # else:
    #     j = 2
    # if 'MCAR' in data_type:
    #     i = 0
    # else:
    #     i = 1

    # for k, prop_latent in enumerate([0.3, 0.7]):
        # data = scores_no_depth.query('prop_latent == @prop_latent')
        # Compute relative scores to Bayes Predictor
    data_relative = scores_no_depth.copy().set_index('method')
    data_relative['R2_test'] = data_relative.groupby(
        ['iter', 'n', 'prop_latent'])['R2_test'].transform(
            lambda df: df - df["BayesPredictor"])
    data_relative = data_relative.reset_index()
    data_relative = data_relative.query('method != "BayesPredictor"')
    # data_relative['method'] = data_relative['method'].map(
    #     NAMES)

    # For real data where the is no prop_latent variable
    # ------------------------------------------------
    data_relative = scores_no_depth.copy().set_index('method')
    data_relative['R2_test'] = data_relative.groupby(
        ['iter', 'n'])['R2_test'].transform(
            lambda df: df - df["GBRT"])
    data_relative = data_relative.reset_index()

    for i, n in enumerate(n_sizes):
        data = data_relative.query('n==@n')
        # ax = axes[i, j+k]
        ax = axes[i]
        ax.grid(axis='x')
        ax.set_axisbelow(True)
        # plot_one(ax, data_relative, NAMES.values(), is_legend=False,
        #             dim=(data_size == '_2500'))
        plot_one(ax, data, methods=None, colors=colors, is_legend=False)
        #if 'gaussian_sm' in data_type and prop_latent == 0.7:
        #    if 'stairs' in data_type:
        #        ax.set_xlim(0.3, 0.7)
        #    elif 'square' in data_type:
            #        ax.set_xlim(0.1, 0.8)

    # for ax in axes[0]:
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, 0.05 * (xmax - xmin))

    # plt.text(0.44, 0, 'Drop in R2 compared to Bayes predictor',
    #          transform=fig.transFigure, va='bottom', size=13)

    # rect = patches.Rectangle((0, .413), width=.085, height=.085, facecolor='k',
    #                          transform=fig.transFigure, zorder=0)
    # fig.add_artist(rect)
    # plt.text(0.002, 0.468, 'MNAR',
    #          transform=fig.transFigure, ha='left', size=14,
    #          color='1', zorder=20)
    # plt.text(0.002, 0.421, 'Gaussian\nself masking',
    #          transform=fig.transFigure, ha='left', size=9,
    #          color='1', zorder=20)

    # plt.text(0.002, 0.9, 'MCAR',
    #          transform=fig.transFigure, ha='left', size=14,
    #          color='1', bbox=dict(facecolor='k'))

    # plt.text(0.42, 0.995, 'Bowl',
    #          transform=fig.transFigure, ha='center', va='top', size=13)
    # plt.text(0.807, 0.995, 'Wave',
    #          transform=fig.transFigure, ha='center', va='top', size=13)

    # line1 = lines.Line2D((.243, .593), (.969, .969), color='k',
    #                     transform=fig.transFigure)
    # fig.add_artist(line1)

    # line2 = lines.Line2D((.635, .985), (.969, .969), color='k',
    #                     transform=fig.transFigure)
    # fig.add_artist(line2)


    # axes[0, 0].set_title('high correlation: easy', size=12, pad=2)
    # axes[0, 1].set_title('low correlation: hard', size=12, pad=2)
    # axes[0, 2].set_title('high correlation: easy', size=12, pad=2)
    # axes[0, 3].set_title('low correlation: hard', size=12, pad=2)

    plt.subplots_adjust(left=.24, bottom=.06, right=.998, top=.93,
                        wspace=.1, hspace=.1)
    # plt.savefig('../figures/boxplots.pdf'.format(data_type),
    #             edgecolor='none', facecolor='none', dpi=100)

    # plt.close()
    plt.show()
