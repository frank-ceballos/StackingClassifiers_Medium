""" ***************************************************************************
# * File Description:                                                         *
# * An example of how to stack classifiers using a synthetic dataset.         *
# *                                                                           *
# * The contents of this script are:                                          *
# * 1. Importing Libraries                                                    *
# * 2. Get data                                                               *
# * 3. Create train and test set                                              *
# * 4. Classifiers                                                            *
# * 5. Hyper-parameters                                                       *
# * 6. Feature Selection: Removing highly correlated features                 *
# * 7. Tuning a classifier to use with RFECV                                  *
# * 8. Custom pipeline object to use with RFECV                               *
# * 9. Feature Selection: Recursive Feature Selection with Cross Validation   *
# * 10. Performance Curve                                                     *
# * 11. Feature Selection: Recursive Feature Selection                        *
# * 12. Visualizing Selected Features Importance                              *
# * 13. Classifier Tuning and Evaluation                                      *
# * 14. Visualing Results                                                     *
# *                                                                           *
# * --------------------------------------------------------------------------*
# * AUTHORS(S): Frank Ceballos <frank.ceballos89@gmail.com>                   *
# * --------------------------------------------------------------------------*
# * DATE CREATED: July 16, 2019                                               *
# * --------------------------------------------------------------------------*
# * NOTES: None                                                               *
# * ************************************************************************"""

%matplotlib auto 
###############################################################################
#                          1. Importing Libraries                             #
###############################################################################
# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

# Classifiers
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingCVClassifier # <- Here is our boy

# Used to ignore warnings generated from StackingCVClassifier
import warnings
warnings.simplefilter('ignore')


###############################################################################
#                                 2. Get data                                 #
###############################################################################
X, y = make_classification(n_samples = 1000, n_features = 30, n_informative = 5,
                           n_redundant = 15, n_repeated = 5, 
                           n_clusters_per_class = 2, class_sep = 0.5,
                           random_state = 1000, shuffle = False)


###############################################################################
#                        3. Create train and test set                         #
###############################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,
                                                    random_state = 1000)


###############################################################################
#                               4. Scale data                                 #
###############################################################################
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


###############################################################################
#                               5. Classifiers                                #
###############################################################################
# Initializing Ada Boost classifier
classifier1 = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 5),
                                 n_estimators = 200,
                                 learning_rate = 0.5,
                                 random_state =1000)

# Initializing Gradient Boosting classifier
classifier2 = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 6,
                                         max_features = "auto", min_samples_leaf = 0.005,
                                         min_samples_split = 0.005, n_estimators = 200,
                                         subsample = 0.8,random_state =1000)

# Initialing Random Forest classifier
classifier3 = RandomForestClassifier(n_estimators = 500, criterion = "gini", max_depth = 10,
                                     max_features = "auto", min_samples_leaf = 0.05,
                                     min_samples_split = 0.01, n_jobs = -1,random_state = 500)

# Initializing the StackingCV classifier
sclf = StackingCVClassifier(classifiers = [classifier1, classifier2, classifier3],
                            use_probas = False,
                            meta_classifier = AdaBoostClassifier())





# Initializing Ada Boost classifier
classifier1 = AdaBoostClassifier()

# Initializing Gradient Boosting classifier
classifier2 = GradientBoostingClassifier()

# Initialing Random Forest classifier
classifier3 = RandomForestClassifier()

# Initializing the StackingCV classifier
sclf = StackingCVClassifier(classifiers = [classifier1, classifier2, classifier3],
                            use_probas = False,
                            meta_classifier = AdaBoostClassifier())



# Create list to store classifiers
classifiers = {"AdaBoost": classifier1,
               "XGB": classifier2,
               "RF": classifier3, 
               "Stacking": sclf}



###############################################################################
#                               6. Train classifiers                          #
###############################################################################
# Train classifiers
for key in classifiers:
    # Get classifier
    classifier = classifiers[key]
    
    # Fit classifier
    classifier.fit(X_train, y_train)
        
    # Save fitted classifier
    classifiers[key] = classifier
  
    
###############################################################################
#                              7. Making predictions                          #
###############################################################################
# Get results
results = pd.DataFrame()
for key in classifiers:
    # Make prediction on test set
    y_pred = classifiers[key].predict_proba(X_test)[:,1]
    
    # Save results in pandas dataframe object
    results[f"{key}"] = y_pred

# Add the test set to the results object
results["Target"] = y_test
    

###############################################################################
#                              8. Visualzing results                          #
###############################################################################
# Probability Distributions Figure
# Set graph style
sns.set(font_scale = 1.5)
sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
               'ytick.color': '0.4'})

# Plot
f, ax = plt.subplots(figsize=(13, 4), nrows=1, ncols=4)

for key, counter in zip(classifiers, range(4)):
    # Get predictions
    y_pred = results[key]
    
    # Get AUC
    auc = metrics.roc_auc_score(y_test, y_pred)
    textstr = f"AUC: {str(auc)}"

    
    false_pred = results[results["Target"] == 0]
    sns.distplot(false_pred[key], hist=True, kde=False, 
                 bins=int(25), color = 'red',
                 hist_kws={'edgecolor':'black'}, ax = ax[counter])
    
    true_pred = results[results["Target"] == 1]
    sns.distplot(results[key], hist=True, kde=False, 
                 bins=int(25), color = 'green',
                 hist_kws={'edgecolor':'black'}, ax = ax[counter])
    
    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[counter].text(0.05, 0.95, textstr, transform=ax[counter].transAxes, fontsize=14,
                    verticalalignment = "top", bbox=props)
    
    # Set axis limits and labels
    ax[counter].set_title(f"{key} Distribution")
    ax[counter].set_xlim(0,1)
    ax[counter].set_xlabel("Probability")

# Tight layout
plt.tight_layout()