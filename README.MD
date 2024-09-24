# Analyzing FiveThirtyEight's Ultimate Halloween Candy Power Ranking Dataset

For a description of the dataset, s. [here](https://fivethirtyeight.com/videos/the-ultimate-halloween-candy-power-ranking/).

## Folders and Scripts

### Folders:
- **`./`**: contains the scripts
- **`data`**: raw data; including a record of the taken survey and pictures of
presented candies over the course of the survey.
- **`results`**: figures (mostly not pretty but they do their job) and (text-based)
results of the modeling procedure.

### Scripts
- **`inspect_survey_stimuli.py`**: Downloads pictures of candies using urls
collected during a test run of the survey
(s. [here](https://walthickey.com/2017/10/18/whats-the-best-halloween-candy/));
- **`eda_text_based.py`**:  Downloads the [dataset](https://raw.githubusercontent.com/fivethirtyeight/data/master/candy-power-ranking/candy-data.csv);
performs some exploratory data analysis, e.g.
calculating descriptive statistics, and prints them to console; also, contains
some thoughts and considerations in the comments regarding results and
how to proceed.
- **`eda_plot_based.py`**: Performs further exploratory data analysis and creates
plots; most importantly, it performs a hierarchical clustering and plots the
results in 2D space (after having performed a dimensionality reduction using
Multiple Correspondence Analysis); also, contains some thoughts and
considerations in the comments.
- **`predictive_modeling.py`**: Performs model fit and model evaluation on the
datasets (imo, overkill in the current case but I did it anyway); further, the
script searches for a combination of product features with a high win
percentage (aka "design a popular candy"); also, the usual considerations per
step.
- **`setup_venv.sh`**: sets up a python virtual environment to easily rerun the
scripts on different machines.


## Setting Up the Virtual Environment

In case you do not want to just take a look at the scripts and figures but also
rerun the scripts (to e.g., see the console ouputs), here are the instructions
to recreate the python virtual environment; everything is hard-coded, so the
scripts do not expect parameters; just rerun...

### Prerequisites

1. **Python**: Ensure that Python3 is installed on your system.
2. **pip**: Ensure that `pip` is installed and up-to-date.

### Setup Instructions

Make the script `setup_venv.sh` executable and run it to create and set up the
new virtual environment:

```
chmod +x setup_venv.sh
./setup_venv.sh <new-env-path>
```

### Activate the new environment:

```
source <new-env-path>/bin/activate
```
