#!/usr/bin/env python3
'''
created on Sunday, September 8, 2024
author: Christian Olaf Haeusler

This script is downloads the datasets and prints out some raw data and summary
statistics to allow a first understanding of the dataset
'''


# imports
import os
import pandas as pd
import requests
from urllib.parse import urlsplit, unquote


# constants
DATASET_URL = 'https://raw.githubusercontent.com/fivethirtyeight/data/master/'\
    'candy-power-ranking/candy-data.csv'
OUTDIR = 'data'


def get_filename_from_url(url):
    '''extracts the filename from a URL.'''
    path = urlsplit(url).path
    return os.path.basename(unquote(path))


def download_pics(url, save_directory):
    # create the directory if it does not exist
    os.makedirs('data', exist_ok=True)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/58.0.3029.110 Safari/537.3'
    }

    # extract the filename from the URL
    filename = get_filename_from_url(url)
    image_path = os.path.join(save_directory, filename)

    # check if the image already exists
    if not os.path.exists(image_path):

        try:
            # get the image data from the URL with headers
            response = requests.get(url, headers=headers)
            # check for HTTP errors
            response.raise_for_status()

            # save the image to the directory
            with open(image_path, 'wb') as file:
                file.write(response.content)

            print(f'Downloaded {url} to {image_path}')
        except Exception as e:
            print(f'Failed to download {url}: {e}')

    return None


def download_csv(url, save_directory):
    '''Downloads a CSV file from the given URL and saves it to the specified
    directory.'''
    # Create the directory if it does not exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Extract the filename from the URL
    fname = os.path.basename(url)
    save_path = os.path.join(save_directory, fname)

    if os.path.exists(save_path):
        print(f'Skipped {url}: {save_path} already exists.\n')

    else:
        try:
            # Send a request to get the CSV file
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Save the CSV file to the specified path
            with open(save_path, 'wb') as file:
                file.write(response.content)

            print(f'Downloaded {url} to {save_path}\n')
        except Exception as e:
            print(f'Failed to download {url}: {e}')

    return fname


if __name__ == "__main__":
    # download the load the dataset
    # download_dataset(DATASET_URL, OUTDIR)
    filename = download_csv(DATASET_URL, OUTDIR)

    # load the dataset into ram
    raw_df = pd.read_csv(os.path.join(OUTDIR, filename))

    # print head, tail and shape
    print(raw_df)
    # general thoughts:
    # a) it's not the raw data with all pairwise comparisons but
    # summary statistics (esp. winpercent) across participants and product
    # b) nice would have been weight (gr), sugar (gr), absolute price
    # but what you gonna do...
    # c) brand / producer are often missing; and no home brand's of retail
    # stores at all? that would probably THE variable to explore...
    # d) 9 dichotomous variables that are more often 0 than 1 for each sample
    # -> might get important later for the model
    # e) 'pricepercent': check if calculated per package (pluribus) or per
    # item; might be per item (despite packages are often shown in the pic?)
    # f) pricepercent and winpercent of 'One quarter' is pretty similiar :-),
    # dime and quarter are now candy obviously, but keep them for now; in case,
    # I will cluster the data, they and just they should be in a cluster

    # most importantly:
    # to build a model that predicts win percentage, there are 12 potential
    # independent variables and just 85 (-2 coins) samples!
    # the biggest question:
    # how to handle that the low amount of data samples?
    # some ideas:
    # - theory-driven feature selection (most obv.: do not use competitorname),
    # - exploratory data analysis (EDA), esp. correlation matrix, might help
    # (done on the whole, ie. non-split, dataset which is a little data
    # snooping but should be "acceptable")
    # - given an extensive EDA, the subsequent ML might not even give more
    # insights (a.k.a. just because you could do ML, does not mean you
    # neccesarilly should do ML)
    # - better avoid data-driven feature selection (elemination methods in case
    # of regression models; ridge or lasso regression need hyper-parameter
    # tuning, further splitting the dataset)
    # - maybe, transforming win percentage into an ordinal variable to predict,
    # makes sense?
    # -> check that, if that makes sense at all
    # - dimensionality reduction (like PCA or EFA)?
    # - use stratified k-fold cross-validation (done before [!] dim. red.!)?
    # - use simple but biased models
    # - use ensemble models (bagging, boosting)?

    # check Dtypes and for missing values
    print(raw_df.info(), '\n')

    print('Balance of dichotomous features')
    # print(raw_df.iloc[:, 1:10].sum(), '\n')
    print(raw_df[raw_df.columns[1:-3]].agg(['sum','count']), '\n')
    # thought:
    # a) unbalanced and mostly zero
    # b) chocolate and fruity are common features
    # -> but they do not seem to occur in combination in the same product
    # -> clustering based on features might give more insight

    # check descriptive statistics;
    # most interesting will be the interval scaled variables
    # but take all columns; maybe, there is something obivously odd in the
    # dichotomous
    print(raw_df.describe().T, '\n')
    # general thoughts:
    # a) mean of interval scaled variables is near 50
    # -> don't worry so much about it, since histograms and density will be
    # plotted anyway

    # show the top and bottom 10 items per columns sugarpercentile,
    # pricepercentile, and win percentage to get intuitive understanding of
    # products
    pd.set_option('display.min_rows', 20)

    print('Winners and losers:')
    print(raw_df.sort_values('winpercent', ascending=False), '\n')
    # toughts:
    # a) chocolate is not just common positively correlated with win percentile
    # b) no fruity in top 10
    # c) peanuty is popular, too (in the U.S., prevalence of peanut allergy in
    # general population (children and adults) is about 1-2%,
    # almond allergy 0.2-0.5%)
    # d) 9 of top 10 are all in price percentile >50
    # -> plot candy ordered by win percentile

    print('High vs. low sugar:')
    print(raw_df.sort_values('sugarpercent', ascending=False), '\n')
    # toughts:
    # imo, no clear trends; maybe the plots will give a better picture

    print('High vs. low price:')
    print(raw_df.sort_values('pricepercent', ascending=False))
    # thoughts:
    # a) chocolate (bars) tend to be more on the expensive side but positively
    # correlated with win percentile
    # b) my assumption was that peanuty would tend to be more expensive and
    # crispedricewafer would tend to be more affordable; but that is not
    # supported by visual inspection. I learned something today, which is nice

    # clean up
    pd.reset_option('display.min_rows')
