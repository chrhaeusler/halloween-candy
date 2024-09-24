#!/usr/bin/env python3
'''
created on Friday, September 6, 2024
author: Christian Olaf Haeusler

To Do:
- cleaning of unecessary functions
- use snake_case for variables

This script is meant to provide materials, i.e. the pictures of candies, to
help the analysist to famliarize her-/himself with the process (i.e. online
survey) that generated the data of the FiveThirtyEight's article
"The Ultimate Halloween Candy Power Ranking".
(s. http://fivethirtyeight.com/features/the-ultimate-halloween-candy-power-ranking/)

The script reads in urls of pictures that were collected during a test run of
the survey, filters them and downloads the pictures. The collected pictures
are quickly accesible when results of the subsequent exploratory data analysis
and calculated models need to be interpreted.
'''

# imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
from urllib.parse import urlsplit, unquote
from pprint import pprint

# constants
fileWithUrls = 'data/candies-presented.txt'
outDir = 'data/pics'


def get_filename_from_url(url):
    """extracts the filename from a URL."""
    path = urlsplit(url).path
    return os.path.basename(unquote(path))


def download_images(urls, saveDir):
    # create the directory if it does not exist
    os.makedirs(saveDir, exist_ok=True)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    for url in urls:
        # extract the filename from the URL
        filename = get_filename_from_url(url)
        imagePath = os.path.join(saveDir, filename)

        # check if the image already exists
        if os.path.exists(imagePath):
            # print(f"Skipped {url}: {filename} already exists.")
            continue

        try:
            # get the image data from the URL with headers
            response = requests.get(url, headers=headers)
            # check for HTTP errors
            response.raise_for_status()

            # save the image to the directory
            with open(imagePath, 'wb') as file:
                file.write(response.content)

            print(f"Downloaded {url} to {imagePath}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")



if __name__ == "__main__":
    # Familiarize yourself with the survey design
    # s. https://walthickey.com/2017/10/18/whats-the-best-halloween-candy/
    # and the pics / stimuli and products
    # results:
    # presented urls of pics are chronolgically recorded in
    # data/candies-presented.txt

    # general thoughts on the survey:
    # behavioral data >> what people say (even when people try to answer
    # honestly)!
    # the design is "worthy of improvement":
    # a) probably high selection bias of participants & demographic data of
    # participants are not even asked
    # b) the idea of offering money is, in principle, good but there should be
    # also an amount that clearly is more valuable than any sweety (e.g., to
    # catch participants that quickly / randomly click on offered choices)
    # c) pics of whole packages with items vs. pics with single items might
    # bias selection (more in children/adolescents than in adults?)
    # d) adult participants have to assume what children want
    # (though, they might rely on experiences during former Halloweens)
    # e) the actual buyers and therefore "targets" are the adults, not the
    # children. So, better than "which would you prefer as a trick-or treater"
    # (usually a child) might be "which would you choose to purchase to offer
    # to trick-or-treaters"
    # fun fact: on Halloween last year, I always placed one onion among the
    # sweeties in a bowl; every of the three groups that rang on the door took
    # the one onion additionally to sweeties. but anyway...
    # d) Variables / features are missing, e.g. some items are encoded as just
    # having chocolate and nuts but they are NOT "bars of chocolate" but have
    # another filling (exception Mr. Goodbar)!

    # download the stimuli / pics

    # read in the file providing the urls of presented pics
    lines = []
    with open(fileWithUrls, 'r') as file:
        for line in file:
            # Strip leading/trailing whitespace and check if the line is non-empty
            stripped_line = line.strip()
            # catch possibly empty lines
            if stripped_line:
                lines.append(stripped_line)

    # do some cleaning
    candieSet = set(lines)
    candieList = sorted(candieSet)

    # summarize what you got
    print(len(candieList), 'pics available')

    # print the file names 'cause some products have more than one pic
    candiePicList = [get_filename_from_url(url) for url in candieList]
    pprint(candiePicList)

    # create output directory to save the pics into
    os.makedirs(outDir, exist_ok=True)

    # download the pictures
    download_images(candieList, outDir)
