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

    # Plot test
    # sns.boxplot(
    #    data=data.query('train_test == "test"'), x='r2', order=methods,
    #            saturation=1, y='method', ax=ax, boxprops=dict(alpha=alpha),
    #            whiskerprops=dict(alpha=alpha),
    #            flierprops=dict(alpha=alpha))
    sns.violinplot(
        data=data, x='R2_test', order=methods, saturation=1, y='method', ax=ax,
        scale="width",  color=('.8' if dim else None))

    for i in range(len(methods)):
        if i % 2:
            ax.axhspan(i - .5, i + .5, color='.9', zorder=0)

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
            2, 2, figsize=(7, 6), sharex='col', sharey=True)

    for data_type in ['MCAR_discontinuous_linear',
                      'gaussian_sm_discontinuous_linear']:

        scores = pd.read_csv('../results/' + data_type + '.csv', index_col=0)
        # scores_GBRT = pd.read_csv(
        #     '../results/' + data_type + '_GBRT.csv', index_col=0)
        # scores = scores.query('method != "GBRT"')
        # scores_GBRT = scores_GBRT.query('method != "BayesPredictor" and ' +
        #                                 'method != "BayesPredictor_order0"')
        # scores = pd.concat([scores, scores_GBRT], axis=0, join='outer')
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
        # Averaging over iterations
        mean_score = scores_no_na.groupby(
            ['method', 'n', 'prop_latent', 'depth', 'mlp_depth', 'lr',
             'weight_decay', 'width_factor'])['R2_val'].mean()
        mean_score = mean_score.reset_index()
        mean_score = mean_score.sort_values(
            by=['method', 'n', 'prop_latent', 'R2_val'])
        best_depth = mean_score.groupby(
            ['method', 'n', 'prop_latent']).last()[
                ['depth', 'mlp_depth', 'lr', 'weight_decay', 'width_factor']]
        best_depth = best_depth.rename(
            columns={'depth': 'best_depth', 'mlp_depth': 'best_mlp_depth',
                     'lr': 'best_lr', 'weight_decay': 'best_weight_decay',
                     'width_factor': 'best_width_factor'})
        scores_no_na = scores_no_na.set_index(
            ['method', 'n', 'prop_latent']).join(best_depth)
        scores_no_depth = scores_no_na.reset_index()
        tmp = ('depth == best_depth and mlp_depth == best_mlp_depth' +
               ' and lr == best_lr and weight_decay == best_weight_decay' +
               ' and width_factor == best_width_factor')
        scores_no_depth = scores_no_depth.query(tmp)

        # Correct by the mean performance across methods
        data_relative = scores_no_depth.copy()
        data_relative['R2_test'] = data_relative.groupby(
            ['iter', 'n', 'prop_latent'])['R2_test'].transform(
                lambda df: df - df.mean())
        data_relative = data_relative.reset_index()
        data_relative = data_relative.query('method != "BayesPredictor"')
        data_relative['method'] = data_relative['method'].map(NAMES)

        j = 0
        if 'MCAR' in data_type:
            i = 0
        else:
            i = 1

        for k, prop_latent in enumerate([0.3, 0.7]):
            for n in [2e4, 1e5]:
                data = data_relative.query(
                    'n == @n and prop_latent == @prop_latent')
            ax = axes[i, j+k]
            ax.grid(axis='x')
            ax.set_axisbelow(True)
            dim = n == 2e4
            plot_one(ax, data, NAMES.values(), is_legend=False, dim=dim)

    plt.text(0.32, 0, 'Change in R2 compared to average across methods',
             transform=fig.transFigure, va='bottom', size=13)

    rect = patches.Rectangle((0, .413), width=.12, height=.085, facecolor='k',
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

    plt.text(0.66, 0.995, 'Discontinuous Linear',
             transform=fig.transFigure, ha='center', va='top', size=13)

    line1 = lines.Line2D((.33, .985), (.969, .969), color='k',
                         transform=fig.transFigure)
    fig.add_artist(line1)

    axes[0, 0].set_title('high correlation: easy', size=12, pad=2)
    axes[0, 1].set_title('low correlation: hard', size=12, pad=2)

    plt.subplots_adjust(left=.33, bottom=.06, right=.996, top=.93,
                        wspace=.1, hspace=.1)
    # plt.show()
    plt.savefig('../figures/boxplots_discontinuous_linear.pdf',
                edgecolor='none', facecolor='none', dpi=100)
    plt.close()
