#!/usr/bin/env python3
'''
created on Sunday, September 7, 2024
author: Christian Olaf Haeusler

To Do:
    - IN CASE OF THE STRIPPLOT USE VIOLIN PLOT!
    - hide color bars (or just use Inkscape...)
    - add a legend to 2d plot

This script creates plots for visual exploratory data analysis
'''

# imports
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import MultipleLocator
import numpy as np
import os
import pandas as pd
import seaborn as sns
import prince  # For Multiple Correspondence Analysis (MCA
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import MinMaxScaler


# constants
INDIR = 'data'
FILENAME = 'candy-data.csv'
OUTDIR = 'results'
PLURIBUS_BOOL = False
CLUSTER_THRESHOLD = 0.55
COLOR_MAP = 'Set1'  # 'viridis'
GERMAN_LABELS = ['Produkt',
                 'Schokolade',
                 'fruchtig',
                 'Karamell',
                 'nussig',
                 'Nougat',
                 'knusprig',
                 'hart',
                 'Riegel',
                 'mehrere',
                 'Zuckerrank',
                 'Preisrank',
                 'Gewinn%'
                 ]


def plot_pairplot(df, corner=True):
    '''
    '''
    g = sns.pairplot(df, corner=corner)

    g.fig.set_size_inches(20, 12)

    plt.tight_layout()

    extensions = ['png', 'svg']
    for extension in extensions:
        fpath = os.path.join(OUTDIR, f'pairplot.{extension}')
        plt.savefig(fpath, bbox_inches='tight')

    plt.show()

    return None


def plot_class_counts(df):
    '''Visualize the imbalance of the dichotomouos variables.

    Maybe, later inspection of correlations provide a hint that we could drop
    a variable that has a low count?'''

    # calculate the sums for the respective columns
    num_samples = df[~df.iloc[:, 0].str.contains('One', na=False)].shape[0]
    counts = df[df.columns[1:-3]].agg(['sum'])

    # quick fix 'cause updating pandas fucked up assumptions of seaborn's
    # barplot function
    pd.DataFrame.iteritems = pd.DataFrame.items

    # create the figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Title and labels
    plt.title(f'Häufigkeit von Produkteigenschaften')
    plt.xlabel('Produkteigenschaft')
    plt.ylabel(f'Anzahl (von N={num_samples})')

    # Do the actual plotting
    sns.barplot(data=counts, ax=ax)

    # draw line for ceiling; substract 2 from 85 'cause these are the couns
    ax.axhline(y=num_samples, color='black', linestyle='--')
    ax.set_ylim([0, 90])
    ax.yaxis.set_major_locator(MultipleLocator(5))

    plt.tight_layout()

    extensions = ['png', 'svg']
    for extension in extensions:
        fpath = os.path.join(OUTDIR, f'counts-features.{extension}')
        plt.savefig(fpath, bbox_inches='tight')

    plt.show()

    return None


def plot_heatmap(df):
    '''TO BE WRITTEN
    '''
    # set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # custom diverging colormap
    cmap = sns.diverging_palette(220, 10, sep=1, as_cmap=True)

    matrix = df.corr()
    # generate a mask for the upper triangle
    mask = np.zeros_like(matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    plt.title('Zusammenhang zwischen Produkteigenschaften\n'
              '(gemessen in Pearson Korrelationskoeffzient)')

    # draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(matrix,
                mask=mask,
                cmap=cmap,
                center=0,
                vmin=-1.0, vmax=1,
                annot=True, annot_kws={'size': 8, 'color': 'k'}, fmt='.1f',
                # linewidths=.5,
                cbar_kws={'shrink': .6}
                )

    # Rotate the tick labels
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    plt.tight_layout()

    extensions = ['png', 'svg']
    for extension in extensions:
        fpath = os.path.join(OUTDIR, f'heatmap-features.{extension}')
        plt.savefig(fpath, bbox_inches='tight')

    plt.show()

    return None


def plot_dendrogramm(df, pluribus=True, threshold=0.55):
    '''TO BE WRITTEN
    '''
    # Calculate the distance between measures
    distance_matrix = sch.distance.pdist(df, metric='hamming')
    # standard (eucclidian) makes no sense in current case
    # Jaccard Distance:
    # Best when only the presence of features is meaningful, often preferred
    # for sparse data.
    # Hamming Distance: Best when equal importance is given to presence and
    # absence.
    # We take Hamming here, since we did some feature selection and reduced the
    # sparcity of the data; if Hamming performs poorly, switch to Jaccard
    linkage = sch.linkage(distance_matrix, method='complete')
    # (Ward is for continuous, normally distributed data)

    # Create the figure
    plt.figure(figsize=(10, 15))

    # Title and labels
    plt.title('Hierarchical Clustering of Candies')
    plt.ylabel('Samples')
    plt.xlabel('Distance')

    # Do the actual plotting
    sch.dendrogram(linkage,
                   labels=df.index,
                   orientation='right',
                   # leaf_rotation=0,
                   leaf_font_size=8,
                   distance_sort=True
                   )

    # Draw a horizontal line at the given threshold's value
    plt.axvline(x=threshold, color='black', linestyle='--')

    # Adjust x limit because some items a very similiar
    plt.xlim(-.01)
    # after further inspection: they have the same values!

    plt.tight_layout()

    extensions = ['png', 'svg']
    for extension in extensions:
        fpath = os.path.join(OUTDIR, f'dendrogram.{extension}')
        plt.savefig(fpath, bbox_inches='tight')

    plt.show()

    # Obtain cluster labels
    cluster_labels = sch.fcluster(linkage, t=threshold, criterion='distance')

    return cluster_labels


def plot_2d_space(df, cluster_labels):
    '''TO BE WRITTEN

    TO DO: Hide tick labels
    '''
    # Perform Multiple Correspondence Analysis (MCA)
    # MCA is an extension of Correspondence Analysis, designed specifically for
    # categorical or binary data. It works by analyzing the patterns in binary
    # data and projecting them onto a lower-dimensional space.
    # Advantages:
    # Specifically designed for binary and categorical data, making it naturally
    # suited for such data typesi. Preserves relationships and associations
    # between binary variables better than PCA.
    # Disadvantages:
    # Less commonly used than PCA, so fewer implementation options and less
    # general familiarity. Can be less interpretable in very high dimensions
    # compared to other methods.
    mca = prince.MCA(n_components=2, random_state=42)
    mca_results = mca.fit_transform(pd.DataFrame(df, columns=df.columns))

    # Convert MCA results to a NumPy array for plotting
    data_2d = mca_results.values

    # Prepare plotting
    # Scale the data from 0 to 1
    # The values are not interpretable anyway but it will help when setting
    # the limits of x and y axis

    scaler = MinMaxScaler()
    data_2d = scaler.fit_transform(data_2d)

    # Create a DataFrame to help with plotting and counting
    df_2d = pd.DataFrame(data_2d,
                         columns=['MCA Component 1', 'MCA Component 2']
                         )

    # Finalize the dataframe
    df_2d['Index'] = df.index
    df_2d.set_index('Index', inplace=True)
    df_2d['Cluster'] = cluster_labels

    # Count occurrences of each coordinate
    coordinate_counts = df_2d.groupby(['MCA Component 1', 'MCA Component 2'])
    coordinate_counts = coordinate_counts.size().reset_index(name='Count')

    # Merge counts with original data
    components_and_clusters = df_2d.merge(
        coordinate_counts,
        on=['MCA Component 1', 'MCA Component 2']
    )

    # Plot the 2D representation with cluster labels and index labels
    plt.figure(figsize=(10, 10))

    features = ', '.join(list(df.columns))
    # Title and labels
    plt.title('Produktgruppen basierend auf Charakteristika'
              f'\n({features})'
              )

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Calculate size of points to be plotted based on count
    sizes = components_and_clusters['Count'] * 100  # scale factor for marker

    # Do the plotting
    plt.scatter(components_and_clusters['MCA Component 1'],
                components_and_clusters['MCA Component 2'],
                c=components_and_clusters['Cluster'],
                cmap=COLOR_MAP,
                s=sizes,
                alpha=1
                )

    # Colorbar
    # better hide it for now, it does not add much value for understanding
    # the plot
    # plt.colorbar(scatter, label='Cluster Label')

    # Annotate the points
    # Vertical offset for annotations the points
    vertical_offset = -0.02
    # Horizontal offset
    horizontal_offset = 0.0
    # the plot needs some final finish in a vector graphic software but anyway

    # Collect annotations by coordinates
    annotations = {}
    for i, (x, y) in enumerate(data_2d):
        # Use rounded coordinates to group nearby points
        key = (round(x, 2), round(y, 2))
        if key not in annotations:
            annotations[key] = []

        annotations[key].append(str(df.index[i]))

    # Add the number of values per point
    for value in annotations.values():
        if len(value) > 3:
            value.insert(0, f'N={len(value)}')

    # Plot annotations
    for key, indices in annotations.items():
        x, y = key
        plt.text(x + horizontal_offset,
                 y + vertical_offset,
                 '\n'.join(indices),
                 fontsize=7,
                 ha='left',
                 va='top',
                 color='black'
                 )

    # Hard coded limits to make the names of all group members clearly visible
    plt.xlim(right=1.11)
    plt.ylim(bottom=-0.14)

    plt.tight_layout()

    extensions = ['png', 'svg']
    for extension in extensions:
        fpath = os.path.join(OUTDIR, f'clusters2d.{extension}')
        plt.savefig(fpath, bbox_inches='tight', transparent=True)

    plt.show()

    return df_2d


def plot_stripplot(samples_info, figsize=(12, 8)):
    '''THIS FUNCTION NEEDS REFACTORING;
    ...and (hindsight bias): better use a violin plot, although the plot will
    get very wide

    Creates a strip plot with adjustable figure size.

    Parameters:
    - data: DataFrame containing the products, their cluster and product
    features
    - figsize: Tuple representing the figure size (width, height).
    '''

    # Prepare plotting of the stripplot
    # Calculate the averages of features per cluster
    averages = samples_info.groupby('Cluster').mean().iloc[:, 2:-3].T
    averages.reset_index(inplace=True)

    # Calculate the average win percentage per cluster
    cluster_order = (
        samples_info.groupby('Cluster')['Gewinn%']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    # Prepare a mapping of categories to average win percentages
    cluster_averages = {
        row['Cluster']: row['Gewinn%']
        for _, row in cluster_order.iterrows()
    }

    # Reshape the data into a long format suitable for plotting
    df_melted = averages.melt(
        id_vars=['index'],
        value_vars=averages.columns[:],
        var_name='Cluster',
        value_name='Averages',
        )

    # Adjust size of the points based on rank of win percentage
    size_mapping = {category: ((len(cluster_order) - rank) * 4.5) ** 2.75
                    for rank, category in enumerate(cluster_order['Cluster'])
                    }

    # Map sizes to the melted data based on categories
    df_melted['PointSize'] = df_melted['Cluster'].map(size_mapping)

    # Create the figure
    plt.figure(figsize=figsize)

    # Title and labels
    plt.title('Durchschnittliches Vorkommen von Eigenschaften je Gruppe\n'
              '(Violinplot, obwohl es sehr breit wird, ist ggf. besser)')
    plt.xlabel('Eigenschaften')
    plt.ylabel('Durchschnittvorkommen')

    # Assign colors to the clusters
    cmap = plt.get_cmap(COLOR_MAP, len(df_melted['Cluster'].unique()))
    cluster_colors = {cluster: cmap(i)
                      for i, cluster
                      in enumerate(df_melted['Cluster'].unique())
                      }

    # Do the plotting
    sns.stripplot(data=df_melted,
                  x='index',
                  y='Averages',
                  hue='Cluster',
                  jitter=False,
                  palette=[cluster_colors[cat] for cat
                           in cluster_order['Cluster']],
                  alpha=0,
                  hue_order=cluster_order['Cluster']
                  )

    # Plot each point manually with sizes based on cluster order
    for i in range(len(df_melted)):
        plt.scatter(
            x=i % len(df_melted['index'].unique()),
            y=df_melted['Averages'].iloc[i],
            s=df_melted['PointSize'].iloc[i],  # size based on cluster order
            color=cluster_colors[df_melted['Cluster'].iloc[i]],
            label='' if i else None,  # Avoid duplicate labels in legend
            alpha=1,  # 1 = no filling
            marker='o',
            facecolor='none',
            linewidth=4
        )

    # Customize the legend
    # Calculate the number of values in each category
    cluster_counts = samples_info['Cluster'].value_counts().to_dict()

    # Create custom legend handles with assigned colors
    custom_handles = [
        patches.Patch(
            color=cluster_colors[cluster],
            label=f'Gruppe {cluster}: '\
            f'{cluster_averages[cluster]:.2f}% avg. win; '\
            f'{cluster_counts[cluster]} Produkte'
        )
        for cluster in cluster_order['Cluster']
    ]

    # Draw the updated legend with custom handles
    plt.legend(handles=custom_handles,
               #  title='Cluster',
               loc='upper right')

    # Adjust the limits
    plt.ylim(-0.1, 1.1)

    plt.grid(True)

    plt.tight_layout()

    extensions = ['png', 'svg']
    for extension in extensions:
        fpath = os.path.join(OUTDIR, f'feat-avg-clusters.{extension}')
        plt.savefig(fpath, bbox_inches='tight')

    plt.show()


def plot_histogram_of_winpercent(df_components_and_features):
    '''TO BE WRITTEN
    '''
    # define some variables for esier use late
    cluster_col = 'Cluster'
    winperc_col = 'Gewinn%'

    # Define clusters
    clusters = df_components_and_features[cluster_col].unique()

    # Set up color palette
    colors = sns.color_palette(COLOR_MAP, len(clusters))

    # Create 2x2 subplot with shared axes
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    plt.suptitle('Verteilung von Gewinn% in den Produktgruppen')
    # Flatten the axes for easier iteration
    axes = axes.flatten()

    for i, cluster in enumerate(clusters):
        cluster_data = df_components_and_features[
            df_components_and_features[cluster_col] == cluster][winperc_col]

        # Calculate statistics
        mean_val = cluster_data.mean()
        median_val = cluster_data.median()
        std_val = cluster_data.std()

        # Calculate number of bins based on the number of members in the cluster
        num_bins = int(np.sqrt(len(cluster_data)))

        # Plot histogram with calculated number of bins
        axes[i].hist(cluster_data,
                     bins=num_bins,
                     alpha=0.7,
                     color=colors[i],
                     edgecolor='black')

        # Plot density curve (KDE)
        # sns.kdeplot(cluster_data, ax=axes[i], color=colors[i], linewidth=2)  # add the KDE line without filling

        # Add title with statistics
        num_members = len(cluster_data)
        axes[i].set_title(f'Cluster {cluster} (N={num_members})\n'
                          f'Mean={mean_val:.2f}, Median={median_val:.2f}, SD={std_val:.2f}')
        axes[i].set_xlabel('Win Percentage')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()

    extensions = ['png', 'svg']
    for extension in extensions:
        fpath = os.path.join(OUTDIR, f'winperc-clusters.{extension}')
        plt.savefig(fpath, bbox_inches='tight')

    plt.show()


def plot_heatmap_per_cluster(df_components_and_features):
    '''TO BE WRITTEN
    '''
    cluster_col = 'Cluster'
    features = df_components_and_features.iloc[:, 3:].columns

    # Define clusters
    clusters = df_components_and_features[cluster_col].unique()

    # Set up color palette
    # colors = sns.color_palette(COLOR_MAP, len(clusters))

    # custom diverging colormap
    cmap = sns.diverging_palette(220, 10, sep=1, as_cmap=True)

    # Create 2x2 subplot for heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    plt.suptitle('Zusammenhang zwischen Variablen in jeder Gruppe\n'
                 '(NaN in case one of the correlated variables is always 0 or 1!)')
    # Flatten the axes for easier iteration
    axes = axes.flatten()

    for i, cluster in enumerate(clusters):
        # Filter data for the current cluster
        cluster_data = df_components_and_features[df_components_and_features[cluster_col] == cluster][features]

        # Compute the correlation matrix
        corr_matrix = cluster_data.corr()

        # generate a mask for the upper triangle
        mask = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        # Plot heatmap
        sns.heatmap(corr_matrix,
                    mask=mask,
                    ax=axes[i],
                    cmap=cmap,
                    center=0,
                    # vmin=-1.05, vmax=1.05,
                    annot=True,
                    fmt='.2f',
                    cbar=True,
                    linewidths=0.5,
                    linecolor='black'
                    )

        # Add title with cluster information
        axes[i].set_title(f'Cluster {cluster}')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0)

    # Adjust layout and show
    plt.tight_layout()

    extensions = ['png', 'svg']
    for extension in extensions:
        fpath = os.path.join(OUTDIR, f'heatmap-feat-clusters.{extension}')
        plt.savefig(fpath, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    # Clean up, in case the script, when run from ipython console, stopped
    # due to a bug and a plot was not closed in the background
    plt.close()

    os.makedirs(OUTDIR, exist_ok=True)
    # Load the dataset into RAM
    raw_df = pd.read_csv(os.path.join(INDIR, FILENAME))

    # Rename all columns, so all plots will show German terms
    raw_df.columns = GERMAN_LABELS

    # Pair plot of all variables
    plot_pairplot(raw_df, corner=True)
    # thoughts:
    # Doesn't tell us much, since most variables are dichtomuous and two
    # interval variables are percentiles; win percent is roughly normally
    # distributed

    # Plot class counts
    plot_class_counts(raw_df)
    # thoughts:
    # most values are zero;
    # we only have 7 candies that contain nougat or rice wafer and their
    # corresponding variables have the lowest amount of variance (cf. tliext
    # based eda); maybe, we can drop these variables, before model fitting; a
    # correlation matrix might be helpful

    # Some cleaning of the dataframe
    # Remove the rows of 'One dime' and 'One quarter'
    df = raw_df[raw_df['Produkt'].str.contains('One') == False]
    # set product name as index
    df.set_index('Produkt', inplace=True)

    # Plot heatmap of linear correlation
    # Pearson correlation is be OK because it is equal to the pointbiserial
    # correlation, which is suitable to examine the relationship between a
    # dichotomous and a continuous variable.
    plot_heatmap(df)
    # thoughts:
    # lots to see, bust most importantly:
    # a) we have linear (!) correlations, yeah!
    # b) with win percentages, highest correlation is chocolate, lowest (most
    # negative) is fruity: chocolate is pos. correlated with bar; most negative
    # correlation is chocolate with fruity;
    # it seems the the most favorite candy is a (peanuty) chocolate bar;
    # in general, chocolate candies might often be single, bigger items,
    # especially when a bar, and fruity candies tend (slightly) to be smaller
    # items from a packages (cf. correlations of chocolate or fruity with
    # pluribus)
    # hence, plotting clusters should be interesting

    # Feature selection before clustering
    # Drop the continuous variables (sugar, price and win percent[ile]);
    # nice side effect: the data are uniformly scaled and we can use a simple
    # distance measure, instead of, e.g., Gower.
    # Drop pluribus, maybe, since it is essentially a measure of item size;
    # however, it is comparatively (negatively) high correlated with win
    # percentage ('cause 'the bigger the better'?)
    # question: is pluribus one single item or can it be, in case of little
    # items, ia bag of bags that is purchased???
    # -> check pictures if values of pluribus are represented in the pics at
    # all

    # drop interval variables, so we have just dichotomouos variables to
    # cluster
    # - sugar percentage has low correlation with win percentage and all
    # product features, anyway
    # - price percentage is probably confounded by 'pluribus': the customer buys
    # the whole packes with many items but the trick-or-treater does not take
    # the whole package but just 1 (or a couple of) item(s)
    # therefore we will get a cluster of product types to better understand the
    # 'candy types' on the market

    if PLURIBUS_BOOL:
        df_feat = df.iloc[:, :-3]
    else:
        df_feat = df.iloc[:, :-4]

    # Drop 'nougat' and 'crispedricewafer', too. They are not very common,
    # and therefore increase the sparcity of the data; moreover, their
    # correlation with Win% is relatively low, too (esp. their partial
    # correlation with Win%?)
    df_feat.drop(['Nougat', 'knusprig'], axis=1, inplace=True)

    # Print candies with unique combination of features
    print('Products with unique combination of features:')
    print(df_feat[~df_feat.duplicated(keep=False)])

    # Perform the hierarchical clustering of products
    # and plot the dendrogramm
    cluster_labels = plot_dendrogramm(df_feat,
                                      pluribus=PLURIBUS_BOOL,
                                      threshold=CLUSTER_THRESHOLD
                                      )

    # Plot the clusters in 2d space after also having performed
    # adimensionality reduction via Multiple Correspondence Analysis
    components_and_clusters = plot_2d_space(df_feat, cluster_labels)

    # Create a new dataframe by merging the original one with the new
    # information we have (values of 2 components and cluster membership of
    # each product)
    df_components_and_features = pd.merge(components_and_clusters,
                                          df,
                                          left_index=True,
                                          right_index=True
                                          )

    # print top winner per cluster
    print('Candies per Cluster ordered by Win%:')
    # Define the column with the four distinct clusters
    cluster_col = 'Cluster'
    # Define the column with the values to sort and print
    winperc_col = 'Gewinn%'

    # Loop through each unique cluster
    for cluster in df_components_and_features[cluster_col].unique():
        # Select rows where the cluster matches and sort by the 'Value' column
        sorted_rows = df_components_and_features[
            df_components_and_features[cluster_col] == cluster].sort_values(by=winperc_col, ascending=False)

        # Print the sorted rows from the second to the last column
        print(f'\nCluster: {cluster}')
        print(sorted_rows.iloc[:, 3:].to_string())

    # Plot the average occurence of features for each cluster to better/more
    # quickly understand of what kind of products with which features the
    # clusters are made of
    # (this plot does not look pretty, but it's sufficient for now)
    # USE VIOLING PLOT!
    plot_stripplot(df_components_and_features, figsize=(15, 6))

    # Interpretation of clusters (ordered by avg. Win%):
    # Cluster 1 (N=11): nutty chocolate (sometimes bar); never caramel or
    # nougat
    # Cluster 2 (N=27): chocolate with caramell (often bar); sometimes nougat
    # or wafer
    # Cluster 4 (N=41): fruity; often 'pluribus' (=small items?); roughly 40%
    # are hard (+some noise: all feat=0 except pluribus)
    # hard
    # Cluster 3 (N=4): Caramell; and essentially nothing else

    # Plot distribution of Win% per cluster
    plot_histogram_of_winpercent(df_components_and_features)
    # thoughts:
    # Cluster 1 has higher variance than Cluster 2 but less members; usually
    # variance gets lower the more samples are drawn from a population; given
    # that less member = less competition; a candy similar to the members in
    # Cluster 1 should be the best option, especially since the distribution
    # seems to be negatively skewed (more candyies with high Win%)

    # TO DO: PLOT HEATMAP OF CORRELATIONS FOR EACH CLUSTER
    plot_heatmap_per_cluster(df_components_and_features)

    # final thoughts after EDA:
    # at lest for now, recommendation to the management:
    # a nutty chocolate candy; but if 'chocolate bar', then something else should be
    # added like caramell or crispyness
    # -> We will use multivariate statistical modeling (aka. machine learning)
    # to figure out if combinations of
    # - bar
    # - nougat
    # - caramell (the correlation of caramell in the heatplot should be
    # attenuated by the caramell-only candies in Cluster 4)
    # - crispyness
    # influences the Win%
    # probably the Win% will increase, the more we add to the candy (all
    # variables are positively correlated with Win%, however
    # often 'less is more'; the finaly decision should be based on further
    # considerations like price, competition, already existing portfolio!

    # in case the amount of data is to low for training/testing (actually, we
    # should test for interactings between variables, too), we can still
    # fall back to a univariate analysis and compare differences between
    # individual features

    # clean up, just in case
    plt.close()
