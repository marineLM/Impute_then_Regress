import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches, lines


def plot_one(ax, data, methods=None, is_legend=False, dim=False):

    if is_legend:
        is_legend = 'full'

    sns.set_palette('bright')

    if methods is None:
        methods = data['method'].unique()

    # sns.violinplot(
    #     data=data, x='R2_train', order=methods, saturation=1, y='method',
    #     ax=ax, scale="width",  color=('.8'))

    sns.violinplot(
        data=data, x='R2_test', order=methods, saturation=1, y='method', ax=ax,
        scale="width",  color=('.8' if dim else None))

    for i in range(len(methods)):
        if i % 2:
            ax.axhspan(i - .5, i + .5, color='.9', zorder=0)

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
    'oracleMLPPytorch': 'Oracle impute + MLP',
    'NeuMiss_uniform_': 'NeuMiss + MLP',
    'MICEMLPPytorch': 'MICE + MLP',
    'MICEMLPPytorch_mask': 'MICE & mask + MLP',
    'meanMLPPytorch': 'mean impute + MLP',
    'meanMLPPytorch_mask': 'mean impute & mask + MLP',
    'GBRT': 'Gradient-boosted trees',
}


if __name__ == '__main__':
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['xtick.major.pad'] = .7
    plt.rcParams['ytick.major.pad'] = 2
    plt.rcParams['ytick.major.size'] = 0
    plt.rcParams['xtick.major.size'] = 2

    fig, axes = plt.subplots(
            2, 4, figsize=(10, 6), sharex='col', sharey=True)

    for data_type in ['MCAR_square', 'MCAR_stairs', 'gaussian_sm_square',
                      'gaussian_sm_stairs']:

        scores = pd.read_csv('../results/' + data_type + '.csv', index_col=0)
        scores_GBRT = pd.read_csv(
            '../results/' + data_type + '_GBRT.csv', index_col=0)
        scores = scores.query('method != "GBRT"')
        scores_GBRT = scores_GBRT.query('method != "BayesPredictor" and ' +
                                        'method != "BayesPredictor_order0"')
        scores = pd.concat([scores, scores_GBRT], axis=0, join='outer')
        # scores = scores.query('n == 100000')
        methods = scores.method.unique()
        methods = methods[methods != 'oracleMLPPytorch_mask']

        # Find the best MLP depth, MLP width, learning rate and weight decay.
        scores_no_na = scores.copy()
        scores_no_na['depth'] = scores_no_na['depth'].fillna(value=0)
        scores_no_na['mlp_depth'] = scores_no_na['mlp_depth'].fillna(value=0)
        scores_no_na['lr'] = scores_no_na['lr'].fillna(value=0)
        scores_no_na['weight_decay'] = scores_no_na['weight_decay'].fillna(
            value=0)
        scores_no_na['width_factor'] = scores_no_na['width_factor'].fillna(
            value=0)
        scores_no_na['max_leaf_nodes'] = scores_no_na['max_leaf_nodes'].fillna(
            value=0)
        scores_no_na['min_samples_leaf'] = scores_no_na[
            'min_samples_leaf'].fillna(value=0)
        scores_no_na['max_iter'] = scores_no_na['max_iter'].fillna(value=0)
        # Averaging over iterations
        mean_score = scores_no_na.groupby(
            ['method', 'n', 'prop_latent', 'depth', 'mlp_depth', 'lr',
             'weight_decay', 'width_factor', 'max_leaf_nodes',
             'min_samples_leaf', 'max_iter'])['R2_val'].mean()
        mean_score = mean_score.reset_index()
        mean_score = mean_score.sort_values(
            by=['method', 'n', 'prop_latent', 'R2_val'])
        best_depth = mean_score.groupby(
            ['method', 'n', 'prop_latent']).last()[
                ['depth', 'mlp_depth', 'lr', 'weight_decay', 'width_factor',
                 'max_leaf_nodes', 'min_samples_leaf', 'max_iter']]
        best_depth = best_depth.rename(
            columns={'depth': 'best_depth', 'mlp_depth': 'best_mlp_depth',
                     'lr': 'best_lr', 'weight_decay': 'best_weight_decay',
                     'width_factor': 'best_width_factor',
                     'max_leaf_nodes': 'best_max_leaf_nodes',
                     'min_samples_leaf': 'best_min_samples_leaf',
                     'max_iter': 'best_max_iter'})
        scores_no_na = scores_no_na.set_index(
            ['method', 'n', 'prop_latent']).join(best_depth)
        scores_no_depth = scores_no_na.reset_index()
        tmp = ('depth == best_depth and mlp_depth == best_mlp_depth' +
               ' and lr == best_lr and weight_decay == best_weight_decay' +
               ' and width_factor == best_width_factor' +
               ' and max_leaf_nodes == best_max_leaf_nodes' +
               ' and min_samples_leaf == best_min_samples_leaf' +
               ' and max_iter == best_max_iter')
        scores_no_depth = scores_no_depth.query(tmp)
        # import IPython
        # IPython.embed()
        # Correct by the Bayes rate
        data_relative = scores_no_depth.copy().set_index('method')
        data_relative['R2_test'] = data_relative.groupby(
            ['iter', 'n', 'prop_latent'])['R2_test'].transform(
                lambda df: df - df["BayesPredictor"])
        data_relative['R2_train'] = data_relative.groupby(
            ['iter', 'n', 'prop_latent'])['R2_train'].transform(
                lambda df: df - df["BayesPredictor"])
        data_relative = data_relative.reset_index()
        data_relative = data_relative.query('method != "BayesPredictor"')
        data_relative['method'] = data_relative['method'].map(NAMES)

        if 'square' in data_type:
            j = 0
        else:
            j = 2
        if 'MCAR' in data_type:
            i = 0
        else:
            i = 1

        for k, prop_latent in enumerate([0.3, 0.7]):
            # for n in [2e4, 1e5]:
            n = 1e5
            data = data_relative.query(
                'n == @n and prop_latent == @prop_latent')
            ax = axes[i, j+k]
            ax.grid(axis='x')
            ax.set_axisbelow(True)
            # dim = n == 2e4
            plot_one(ax, data, NAMES.values(), is_legend=False, dim=False)

    for ax in axes[0]:
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, 0.05 * (xmax - xmin))

    plt.text(0.44, 0, 'Drop in R2 compared to Bayes predictor',
             transform=fig.transFigure, va='bottom', size=13)

    rect = patches.Rectangle((0, .413), width=.085, height=.085, facecolor='k',
                             transform=fig.transFigure, zorder=0)
    fig.add_artist(rect)
    plt.text(0.002, 0.468, 'MNAR',
             transform=fig.transFigure, ha='left', size=14,
             color='1', zorder=20)
    plt.text(0.002, 0.421, 'Gaussian\nself masking',
             transform=fig.transFigure, ha='left', size=9,
             color='1', zorder=20)

    plt.text(0.002, 0.9, 'MCAR',
             transform=fig.transFigure, ha='left', size=14,
             color='1', bbox=dict(facecolor='k'))

    plt.text(0.42, 0.995, 'Bowl',
             transform=fig.transFigure, ha='center', va='top', size=13)
    plt.text(0.807, 0.995, 'Wave',
             transform=fig.transFigure, ha='center', va='top', size=13)

    line1 = lines.Line2D((.243, .593), (.969, .969), color='k',
                         transform=fig.transFigure)
    fig.add_artist(line1)

    line2 = lines.Line2D((.635, .985), (.969, .969), color='k',
                         transform=fig.transFigure)
    fig.add_artist(line2)

    axes[0, 0].set_title('high correlation: easy', size=12, pad=2)
    axes[0, 1].set_title('low correlation: hard', size=12, pad=2)
    axes[0, 2].set_title('high correlation: easy', size=12, pad=2)
    axes[0, 3].set_title('low correlation: hard', size=12, pad=2)

    plt.subplots_adjust(left=.24, bottom=.06, right=.998, top=.93,
                        wspace=.1, hspace=.1)
    # plt.show()
    plt.savefig('../figures/boxplots_neurips2021.pdf',
                edgecolor='none', facecolor='none', dpi=100)

    plt.close()
