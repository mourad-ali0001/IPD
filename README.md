Premier League Match Outcome Prediction

Project Summary

This project implements a machine learning model to predict Premier League match outcomes (Home Win, Draw, Away Win) using features derived from historical football match data.

The focus is primarily on time-aware modelling, feature engineering based on team momentum, and a rigorous comparison of various machine learning models.

The project uses the standard CRISP-DM approach and avoids common pitfalls with the temporal modelling of time-series in sports data sets (e.g., temporal data leakage).


General Goals

- Engineer football features that are indicative of the team’s recent performance and momentum

- Compare different ML models in a rigorous way

- Use time-series cross-validation

- Evaluate the models using the macro F1-score metric

- Give interpretable outcomes that can be reported/visualized


Dataset

The dataset is a collection of historical Premier League matches that contain the following:

- Date of the matches

- Home teams and away teams

- The match outcome (FTR)

- Features (match statistics) such as:

  - Shots on target

  - Corners

  - Fouls

  - Yellow and red cards

The target variable FTR is encoded as:

- 2 → Home Win

- 1 → Draw

- 0 → Away Win

All features are calculated in temporal order so that future data is not used in the creation of features.


Feature Engineering

Feature engineering is a major contribution of the project.

Recent Form (Last 5 Matches)

For each match, features are computed for the home team and away team separately, only using their last five matches:

- Weighted wins, draws, losses

- Weighted points (based on football win mechanics: 3 points for a win, 1 for a draw)

- Recency weighting

Instead, general form is not computed, only momentum.


Rolling Averages of Match Statistics

The following rolling statistics are captured over the last five matches:

- Shots on target

- Corners

- Fouls

- Yellow cards

- Red cards

Home and away teams are handled appropriately to ensure the correct context is applied to each match.

Rows without enough historical data are dropped to avoid issues with imputation.

Train/Test Split

- Matches are split chronologically in an 80/20 train-test split

- shuffle=False to avoid temporal leakage

- Test set will contain matches from the future compared to the training set


Machine Learning Models

The following ML models are applied in the study:

- Decision Tree

- Random Forest

- Gradient Boosting

- CatBoost (with class weights due to drawing imbalance)

- Support Vector Machine (RBF Kernel)

- K Nearest Neighbours (Weighted by distance)

- Neural Network (MLP with scaling + early stopping)

- XGBoost (only if available in the environment)

All models that require scaling are implemented through pipelines to avoid leakage between the train and test data sets.


Cross Validation

To evaluate how well each model performs, I use TimeSeriesSplit (5 folds) to implement cross validation on the training dataset only:

- Each fold gets a different set of training/testing splits, where the training set only has earlier matches

- Model performance is averaged across folds

- Models are ranked using the macro F1 score to ensure equal importance of predictions for Home, Draw, and Away outcomes


Evaluation Metrics

The following metrics are recorded for evaluating the performance of the models:

- Accuracy (across classes)

- Macro F1-score

- Precision and Recall (per class)

- Confusion Matrices

The final evaluation is on an unseen test set to mimic real-world predictions.


Specific Challenges & Limitations

- Predictive performance is limited due to the inherent lack of features in football match datasets

- The ability to correctly predict draws can be very difficult due to class imbalances

- Contextual factors (e.g., player availability) are not factored into predictions

In the future, I could:

- Hyperparameter tune the best performing models for improved accuracy

- Engineer additional features related to momentum (e.g., longer rolling averages)

- Consider external datasets/sources (e.g., betting odds)
