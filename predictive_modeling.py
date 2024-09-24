#!/usr/bin/env python3
'''
created on Saturday, September 14, 2024
author: Christian Olaf Haeusler

This scripts fit a Linear Regression, Ridge Regression, Lasso Regression,
and Random Forest using a k-fold cross-validation to the data;
it prints out model performance measures; and, performs a grid
search across candy features to predict a feature combination with high win%

To Do:
    - NEEDS CLEANING!
    - write what is now just printed to the console to file
'''

# imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# constants
INDIR = 'data'
FILENAME = 'candy-data.csv'
PLURIBUS_BOOL = False
OUTDIR = 'results'
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


def evaluate_models(df, models):
    '''TO BE WRITTEN
    '''
    # Split into train and test
    # First 8 columns are predictors, last column is criterion
    x = df.iloc[:, :8]
    y = df.iloc[:, -1]

    # Define the number of splits for k-fold.
    # In case of 5 folds, we have rougly 64:16 of samples for train & test
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Thoughts...
    # We do not have a lot of data, so I do k-fold cross-validation here

    # Dictionaries to store evaluation metrics and feature importance
    metrics = {name: {'r2': [], 'mse': [], 'rmse': [], 'mae': []}
               for name in models}

    feature_importances = {name: None for name in models}

    # Initialize model storage for each fold
    fold_models = {name: [] for name in models}

    # Perform the k-fold cross-validation
    for name, model in models.items():
        print(f'\n{name}:')

        for fold_idx, (train_index, test_index) in enumerate(kf.split(x)):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # I do not perform a scaling since the appplied modeling algorithms
            # (regression models and random forest) do not assume a normal
            # distribution and all predictors are on the
            # same scale
            # from sklearn.preprocessing import StandardScaler
            # Initialize the StandardScaler
            # scaler = StandardScaler()
            # Fit the scaler only on the training data and transform it
            # x_train = scaler.fit_transform(x_train)
            # Transform the test data using the same scaler
            # x_test = scaler.transform(x_test)

            # Train the model
            model.fit(x_train, y_train)
            # Save model for the fold
            fold_models[name].append(model)

            # Predict on test set
            y_pred = model.predict(x_test)

            # Evaluate model performance for this fold
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)

            # Append scores for each fold
            metrics[name]['r2'].append(r2)
            metrics[name]['mse'].append(mse)
            metrics[name]['rmse'].append(rmse)
            metrics[name]['mae'].append(mae)

            print(f'Fold {fold_idx}: r2={r2:.2f}, mse={mse:.2f}, '
                  f'rmse={rmse:.2f}, mae={mae:2f}')

        # Calculate average metrics across all folds
        avg_r2 = np.mean(metrics[name]['r2'])
        avg_mse = np.mean(metrics[name]['mse'])
        avg_rmse = np.mean(metrics[name]['rmse'])
        avg_mae = np.mean(metrics[name]['mae'])

        std_r2 = np.std(metrics[name]['r2'])
        std_mse = np.std(metrics[name]['mse'])
        std_rmse = np.std(metrics[name]['rmse'])
        std_mae = np.std(metrics[name]['mae'])

        # Print evaluation results
        print(f"\n{name}: Mean and standard deviation across folds:")
        print(f"R-squared: M={avg_r2:.2f}; SD={std_r2:.2f}")
        print(f"Mean Squared Error: M={avg_mse:.2f}; SD={std_mse:.2f}")
        print(f"Root Mean Squared: M={avg_rmse:.2f}; SD={std_rmse:.2f}")
        print(f"Mean Absolute Error: M={avg_mae:.2f}; SD={std_mae:.2f}")

        # Thoughts:
        # Given the size of the datset, the stability across folds is higher
        # than expected.

        # Store feature importance for each model
        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importances[name] = np.abs(model.coef_)
        else:
            feature_importances[name] = np.zeros(x.shape[1])

    # Plot feature importance for each model
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=False)
    fig.suptitle('Feature Importance for Different Models')

    for i, (name, importances) in enumerate(feature_importances.items()):
        ax = axs[i // 2, i % 2]
        # Preserve original feature order
        ax.barh(x.columns[::-1], importances[::-1], align='center'),
        ax.set_title(name)
        ax.set_xlabel('Importance')
        ax.set_ylabel('Features')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    extensions = ['png', 'svg']
    for extension in extensions:
        fpath = os.path.join(OUTDIR, f'modeling-feat-importance.{extension}')
        plt.savefig(fpath, bbox_inches='tight')

    plt.show()

    return fold_models


def predict_best_feat_combination(df, fold_models):
    '''TO BE WRITTEN
    - REFACTOR! this function probably uses lots of global variables
    - averaging the (linear) models is probably overkill, just train on the whole
    dataset...
    '''
    # Averaging linear model coefficients
    # (i.e., only for Linear Regression, Ridge, and Lasso)
    avg_models = {}

    for name, models_list in fold_models.items():
        if 'Regression' in name:  # Linear models
            avg_coef = np.mean([model.coef_ for model in models_list if hasattr(model, 'coef_')], axis=0)
            avg_intercept = np.mean([model.intercept_ for model in models_list], axis=0)
            avg_models[name] = {'coef': avg_coef, 'intercept': avg_intercept}

    # Define column names for the grid search (everything except chocolate and
    # fruity
    columns_for_grid = GERMAN_LABELS[3:-4]

    # Create a DataFrame for all combinations of columns 3 to 8
    combinations = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 6)

    # Create the final dataframe
    # always use chocolate, never use fruity
    predictors = pd.DataFrame(combinations, columns=columns_for_grid)
    predictors.insert(0, 'Schokolade', 1)
    predictors.insert(1, 'fruchtig', 0)

    # Create a dataframe to store predictions separately
    prediction_df = predictors.copy()

    for name, avg_model in avg_models.items():
        # Re-slice the original predictors df to ensure no extra columns in
        # loops > 1
        predictor_slice = predictors[GERMAN_LABELS[1:-4]]
        predicted_values = np.dot(predictor_slice.values, avg_model['coef']) + avg_model['intercept']

        # Add the prediction column for this model
        prediction_df[f'Prediction {name}'] = predicted_values

        # Sort by predicted values (highest to lowest)
        sorted_df = prediction_df.sort_values(by=f'Prediction {name}', ascending=False)

        # Display sorted predictions
        print(f"\nPredictions with top win% of {name}:")
        print(sorted_df[[f'Prediction {name}'] + GERMAN_LABELS[1:-4]].head())

    # For Random Forest, simply use the the model from the first fold
    first_random_forest = fold_models['Random Forest'][0]
    # Again, use only the predictor slice
    predictor_slice = predictors[df.columns[:-1]]  # Use only predictors
    prediction_df['Prediction Random Forest'] = first_random_forest.predict(predictor_slice)

    # Sort Random Forest predictions
    sorted_rf_df = prediction_df.sort_values(by='Prediction Random Forest', ascending=False)

    # Display sorted Random Forest predictions
    print("\nPredictions with top win% of Random Forest:")
    print(sorted_rf_df[['Prediction Random Forest'] + GERMAN_LABELS[1:-4]].head())

    # toughts:
    # okay, it's not "the more ingredients, the better"

    # Averaging across results of the algorithms
    # do not take into account the random forest model, since it is the best
    # model across folds and not an average model

    # Add a new column for the average of the predictions
    prediction_df['Average_Prediction'] = prediction_df.iloc[:, -3:-1].mean(axis=1)
    sorted_df = prediction_df.sort_values(by='Average_Prediction', ascending=False)

    # Define the algos to be used
    # hacky, but whatever...
    prediction_columns = [
        'Prediction Linear Regression',
        'Prediction Ridge Regression',
        'Prediction Lasso Regression',
        'Prediction Random Forest'
    ]

    # Printing the results to console
    # Initialize a dictionary to store the results for each algorithm
    means_dict = {}

    # Loop through each prediction column, sort by it,
    # and compute the mean for the top 5 rows
    for col in prediction_columns:
        # Sort the DataFrame by the current prediction columnin descending
        # order
        sorted_df = prediction_df.sort_values(by=col, ascending=False)

        # Calculate the mean of the 3 prediction with highest win%
        # for the current algorithm
        mean_of_top = sorted_df.head(3).mean()

        # Store the result in the dictionary
        means_dict[col] = mean_of_top

    # Create a DataFrame from the dictionary with algorithms as rows
    means_df = pd.DataFrame(means_dict).T

    # Step 1: Calculate the mean of each column across all 4 algorithms
    overall_mean = means_df.mean()
    # Step 2: Create a binary row where
    # values >= 0.5 are set to 1, and values < 0.5 are set to 0
    binary_row = (overall_mean >= 0.5)

    # Step 3: Append both rows (mean and binary row) to the original DataFrame
    final_df = pd.DataFrame([overall_mean, binary_row],
                            index=['Mean', 'Binarized'])

    # Append the results to the original means_df
    final_df = pd.concat([means_df, final_df])

    # Display the final DataFrame with the added mean and binary rows
    print('\nMean of top 3 combinations of features per algorithm, '
          'mean across algos, and binarized final results:')
    print(final_df.iloc[:, :-5].round(decimals=2))

    return None


if __name__ == "__main__":
    # Clean up, in case the script, when run from ipython console, stopped
    # due to a bug and a plot was not closed in the background
    plt.close()

    # Load the dataset into RAM
    raw_df = pd.read_csv(os.path.join(INDIR, FILENAME))

    # Rename all columns, so all plots will show German terms
    raw_df.columns = GERMAN_LABELS

    # Some cleaning of the dataframe
    # Remove the rows of 'One dime' and 'One quarter'
    df = raw_df[raw_df['Produkt'].str.contains("One") == False]
    # set product name as index
    df.set_index('Produkt', inplace=True)

    if not PLURIBUS_BOOL:
        df.drop(['mehrere'], axis=1, inplace=True)

    # drop 'Zuckerrank' und 'Preisrank'
    df.drop(['Zuckerrank', 'Preisrank'], axis=1, inplace=True)

    # change dtype just in case
    for col in GERMAN_LABELS[1:-4]:
        df[col] = df[col].astype('category')

    # shuffle the dataframe, since items are sorted alphabetically which might
    # be correlated with features
    df = df.sample(frac=1, random_state=42)

    # Initialize models
    # since the dataset is so small, let's not do a hyperparameter tuning and
    # consider the modeling as a proof of concept
    # Let's keep it simple here, and worry about Gradient Boosting another time

    # tbh, alpha values for Ridge and Lasso are snooped (just a little bit...)
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=2),
        "Lasso Regression": Lasso(alpha=0.3),
        "Random Forest": RandomForestRegressor(n_estimators=100,
                                               random_state=42)
        }

    # Model evaluation
    # Perform fitting of Multiple, Ridge and Lasso Regression, and Random
    # Forest on k-fold cross-validation
    # print performance measures to console and plot feature importance
    fold_models = evaluate_models(df, models)

    # Design a new candy with high win%
    # From here on, it gets extremely "hacky"
    # Simpler would be to just train on the whole dataset but I chose to take
    # the average model (of the linear models)

    # Initial thought:
    # Probably, the candy with the highstes win% will the one with
    # "the more features that are (somewhat) positively correlated with win%,
    # the better"
    # for now, the scripts only prints the results to console
    predict_best_feat_combination(df, fold_models)

    # Result:
    # Combination of features is the same as in, and only in, Snickers Crysper
    # a chocolate, caramel, nutty, crispy bar
    # which does have a Win% of 60 (and is discontinued, lol!).

    # but hey, it's the only candy with that combination and there for the only
    # competition (at least in the given set of candies)
    # but why?

    # tbh, I did not calculate interactions between variables, to not
    # increase the numbers of predictors given the sparcity of the dataset

    # but if it has to be that combination, because "AI" says so,
    # design a new product that has:
    # - one layer of caramell with shredded nuts, on top of...
    # - another layer comprising a wafel

    # in any case:
    # variables not included in the given dataset must be considered when
    # making a decision

    # imo, the best thing is:
    # take the (nutty chocolate) product that sells the most, copy it,
    # and offer it at a lower price

    # in the current case, exploratory data analysis is sufficient imo,
    # data science/ML is over the top
