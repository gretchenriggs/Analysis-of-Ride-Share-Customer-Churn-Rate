from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data split, normalizing, metrics:
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import partial_dependence
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

#Classifiers:
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


from sklearn.metrics import confusion_matrix, precision_score, recall_score



def SVC_tester(X_train, y_train, X_test):
    K = ['rbf', 'linear', 'sigmoid', 'poly']
    C = [0.01, 0.1, 1, 10]
    for k in K:
        print "================="
        print k, "\n"

        for c in C:
            regrSVC = SVC(C=c, kernel = k)
            regrSVC.fit(X_train, y_train)
            SVC_y_pred = regrSVC.predict(X_test)
            print "---------"

            print "C: ", c, "\nSVC f1 score: ", f1_score(y_test, SVC_y_pred), "\n\n"

class AdaBoostBinaryClassifier(object):
    '''
    INPUT:
    - n_estimator (int)
      * The number of estimators to use in boosting
      * Default: 50

    - learning_rate (float)
      * Determines how fast the error would shrink
      * Lower learning rate means more accurate decision boundary,
        but slower to converge
      * Default: 1
    '''

    def __init__(self, n_estimators=50):

        self.base_estimator = DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators

        # Will be filled-in after fit() called
        self.estimators_ = []
        self.estimator_weight_ = np.zeros(self.n_estimators, dtype=np.float)

    def fit(self, X, y):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix
        - y: numpy array, labels

        Build the estimators for the AdaBoost estimator.
        '''

        sample_weight = np.ones(len(y))
        for i in range(self.n_estimators):
            est, sample_weight, alpha = self._boost(X, y, sample_weight)
            self.estimators_.append(est)
            self.estimator_weight_[i] = alpha


    def _boost(self, X, y, sample_weight):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix
        - y: numpy array, labels
        - sample_weight: numpy array

        OUTPUT:
        - estimator: DecisionTreeClassifier
        - sample_weight: numpy array (updated weights)
        - estimator_weight: float (weight of estimator)

        Go through one iteration of the AdaBoost algorithm. Build one estimator.
        '''

        estimator = clone(self.base_estimator)
        # Note that the .fit and .predict methods below are methods associated
        # with the DecisionTreeClassifier, not the fit and predict methods
        # within this class
        estimator.fit(X, y, sample_weight = sample_weight)
        predictions = estimator.predict(X)
        error = sample_weight[predictions != y].sum()/sample_weight.sum()
        alpha = np.log((1- error)/error)
        new_weights = sample_weight * np.exp(alpha*(predictions!=y).astype(int))

        return estimator, new_weights, alpha


    def predict(self, X):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix

        OUTPUT:
        - labels: numpy array of predictions (0 or 1)
        '''
        # predictions from DecisionTreeClassifier will be 0 and 1
        pred_0_1 = [estimator.predict(X) for estimator in self.estimators_]

        def recode(y):
            '''
            function to change y from {0,1} to {-1,1}
            '''
            y[y==0]=-1
            return y

        # Step 3 of AdaBoost.M1 algorithm requires -1 and 1 predictions
        pred_neg1_1 = map(recode, pred_0_1)
        alpha_Gm = zip(self.estimator_weight_, pred_neg1_1)
        G = np.sign(sum([alpha * Gm for alpha, Gm in alpha_Gm]))
        return (G > 0).astype(int) # y_test is 0 and 1, so change back

    def score(self, X, y):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT:
        - score: float (accuracy score between 0 and 1)
        '''
        preds = self.predict(X)
        y = y.astype(int)
        return float((y==preds).sum())/len(y)

def part_1(X_train, X_test, y_train, y_test):
    '''
    Part 1 of the pair assignment
    Implementing your own AdaBoost Classifier
    '''
    my_ada = AdaBoostBinaryClassifier(n_estimators=50)
    my_ada.fit(X_train, y_train)
    print "\nPart 1 - Implementing your own AdaBoostClassifier"
    print "-" * 50
    print " Custom ABC accuracy: {0:0.3f}".format(my_ada.score(X_test, y_test))

    #Compare to sklearn implementation
    abc = AdaBoostClassifier(n_estimators=50)
    abc.fit(X_train, y_train)
    print "sklearn ABC accuracy: {0:0.3f}".format(abc.score(X_test, y_test))

def misclassification_rate(y_pred, y):
    '''
    INPUT:
        - y: numpy array, true labels
        - y_pred: numpy array, predicted labels
    '''
    return float((y_pred != y).sum())/len(y)

def stage_score_plot(estimator, X_train, X_test, y_train, y_test):
    '''
        Parameters: estimator: GradientBoostingClassifier or
                               AdaBoostClassifier
                    X_train: 2d numpy array
                    y_train: 1d numpy array
                    X_test: 2d numpy array
                    y_test: 1d numpy array

        Returns: A plot of the number of iterations vs the misclassification rate
        for the model for both the training set and test set.
    '''
    estimator.fit(X_train, y_train)
    name = estimator.__class__.__name__.replace('Classifier', '')
    label_str = name
    if "Gradient" in name:
        md = estimator.max_depth
        label_str += ", max depth: {0}".format(md)

    # initialize
    test_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    # Get test score from each boost
    for i, y_test_pred in enumerate(estimator.staged_predict(X_test)):
        test_scores[i] = misclassification_rate(y_test, y_test_pred)
    plt.plot(test_scores, alpha=.5, label=label_str, ls = '-', marker = 'o', ms = 3)
    plt.ylabel('Misclassification rate', fontsize=14)
    plt.xlabel('Iterations', fontsize=14)


def grid_search(X_train, y_train):
    ''' gets a rough idea where the best parameters lie '''
    # note that the number of estimators is set at 100 while the learning rate varies
    boosting_grid_rough = {'learning_rate': np.logspace(-2, 0, num = 3),
                           'max_depth': [1, 3, 10],
                           'min_samples_leaf': [1, 3, 10],
                           'subsample': [1.0, 0.5],
                           'max_features': [None, 'sqrt'],
                           'n_estimators': [100],
                           'random_state': [1]}

    coarse_search = GridSearchCV(GradientBoostingClassifier(), boosting_grid_rough)
    print "\n4) Part 2 grid search"
    print "Starting grid search - coarse (will take several minutes)"
    coarse_search.fit(X_train, y_train)
    coarse_params = coarse_search.best_params_
    coarse_score = coarse_search.best_score_
    print "Coarse search best parameters:"
    for param, val in coarse_params.iteritems():
        print "{0:<20s} | {1}".format(param, val)
    print "Coarse search best score: {0:0.3f}".format(coarse_score)
    # results will vary, but results on this machine:
    #'learning_rate': 0.01
    #'max_depth': 10,
    #'max_features': 'sqrt',
    #'min_samples_leaf': 1,
    #'subsample': 1.0,

    boosting_grid_fine = {'learning_rate': [0.005, 0.01, 0.05],
                          'max_depth': [5, 10, 15],
                          'min_samples_leaf': [1, 2, 3],
                          'subsample': [1.0],
                          'max_features': ['sqrt'],
                          'n_estimators': [100],
                          'random_state': [1]}

    fine_search = GridSearchCV(GradientBoostingClassifier(), boosting_grid_fine)
    print "\nStarting grid search - fine"
    fine_search.fit(X_train, y_train)
    fine_params = fine_search.best_params_
    fine_score = fine_search.best_score_
    print "Fine search best parameters:"
    for param, val in fine_params.iteritems():
        print "{0:<20s} | {1}".format(param, val)
    print "Fine search best score: {0:0.3f}".format(fine_score)
    model_best = fine_search.best_estimator_
    print "Returning best model."
    return model_best

def answer_description_2_3():
    print "3) The Gradient Boosting Classifier that allows for its trees to be 100"
    print "splits deep doesn't perform well with increased iterations (boosts)."
    print "This is expected as boosting helps decrease bias over time, but the"
    print "100 split deep classifier starts with relatively low bias, so boosting"
    print "it will have less effect than it has on the less deep classifiers.  In"
    print "the end, boosting enables the ensemble based on less complex models to"
    print "outperform the boosted, more complex model."

def part_2(X_train, X_test, y_train, y_test):
    '''
    Part 2 of pair assignment
    Investigate estimator complexity
    '''
    print "\nPart 2 - Investigate Estimator Complexity"
    print "-" * 50
    n_trees = 100
    models = [AdaBoostClassifier(n_estimators=n_trees),
              GradientBoostingClassifier(n_estimators=n_trees),
              GradientBoostingClassifier(n_estimators=n_trees, max_depth = 10),
              GradientBoostingClassifier(n_estimators=n_trees, max_depth = 100)
              ]
    for model in models:
        stage_score_plot(model, X_train, X_test, y_train, y_test)
    plt.legend()
    plt.title('Investigate model complexity on Test set', fontsize=14, fontweight='bold')
    plt.grid(color='grey', linestyle='dotted')
    figname = '2_2_model_complexity.png'
    plt.savefig(figname, dpi = 100)
    plt.close()
    print "2) Plot {0} saved.".format(figname)
    answer_description_2_3()
    model_best = grid_search(X_train, y_train)
    # if you don't want to wait for the grid search, comment line above
    # and uncomment lines below
    #model_best = GradientBoostingClassifier(n_estimators = n_trees,
    #                                        learning_rate = 0.05,
    #                                        max_features = 'sqrt',
    #                                        min_samples_leaf = 2,
    #                                        max_depth = 10)
    #model_best.fit(X_train, y_train)
    y_pred = model_best.predict(X_test)
    mc_rate = misclassification_rate(y_pred, y_test)
    print "\n4) The best model misclassification rate is {0:0.3f}".format(mc_rate)
    return model_best

def get_feature_names():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names'
    names = requests.get(url)
    names = names.text.split('\r\n')
    names = itertools.ifilter(lambda x: not x.startswith('|'), names)
    names = [str(name.split(':')[0]) for name in names]
    return np.array(list(names)[4:-1])

def bar_plot(feature_names, feature_importances):
    y_ind = np.arange(9, -1, -1) # 9 to 0
    fig = plt.figure(figsize=(8, 8))
    plt.barh(y_ind, feature_importances, height = 0.3, align='center')
    plt.ylim(y_ind.min() + 0.5, y_ind.max() + 0.5)
    plt.yticks(y_ind, feature_names)
    plt.xlabel('Relative feature importances')
    plt.ylabel('Features')
    figname = '3_1_feature_importance_bar_plot.png'
    plt.tight_layout()
    plt.savefig(figname, dpi = 100)
    plt.close()
    print "\n1) {0} plot saved.".format(figname)

def answer_description_3_2():
    print "\n2) Partial dependency plots show the effect of a given feature on"
    print "the target variable after accounting for the average effect of all"
    print "the other features.  The regions where the graph is flat means"
    print "that there isn't much of a relationship between the target (in"
    print "this case, whether it's spam or not) and the feature investigated"
    print "by that plot in that region.  However, in areas where it isn't flat"
    print "the target is more dependent on the feature in question."
    print "\nAs shown in the plots, as the 'char_freq_!' and 'char_freq_$'"
    print "feature values increase, the chance of spam (value 1) increases"
    print "This is also true for 'capital_run_length_total' for values <"
    print "about 125 characters, however after that it doesn't make much"
    print "difference.  The other plots can be interpreted in a similar way."

def pdp_3d(clf, target_feature, X_train, names):
    fig = plt.figure(figsize = (10, 10))

    ax = Axes3D(fig)

    pdp, (x_axis, y_axis) = partial_dependence(clf, target_feature,
                                               X=X_train, grid_resolution=50)
    XX, YY = np.meshgrid(x_axis, y_axis)
    Z = pdp.T.reshape(XX.shape).T

    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu)
    ax.set_xlabel(names[target_feature[0]])
    ax.set_ylabel(names[target_feature[1]])
    ax.set_zlabel('Partial dependence')
    ax.view_init(elev=22, azim=222)
    plt.colorbar(surf)
    plt.subplots_adjust(top=0.9)
    figname = '3_4_3D_partial_dependency_plot.png'
    plt.savefig(figname, dpi = 100)
    plt.close()
    print "\n4) {0} plot saved.".format(figname)


def part_3(model, X_train):
    '''
    Part 3 of pair assignment
    Feature importance and Partial Dependency Plots
    '''
    print "\nPart 3 - Feature Importance and Partial Dependency Plots"
    print "-" * 50
    # part 1 - feature importance plot
    feature_importances = model.feature_importances_
    top10_colindex = np.argsort(feature_importances)[::-1][0:10]
    feature_importances = feature_importances[top10_colindex]
    feature_importances = feature_importances / float(feature_importances.max()) #normalize

    all_feature_names = get_feature_names()
    feature_names = list(all_feature_names[top10_colindex])

    print "1) Sorted features, their relative importances, and their indices:"
    for fn, fi, indx in zip(feature_names, feature_importances, top10_colindex):
        print "{0:<30s} | {1:6.3f} | {2}".format(fn, fi, indx)

    bar_plot(feature_names, feature_importances)

    # part 2 - plot partial dependence
    plot_partial_dependence(model, X_train, top10_colindex,
                            feature_names = all_feature_names,
                            figsize=(12,10))
    plt.tight_layout()
    figname = '3_2_partial_dependency_plots.png'
    plt.savefig(figname, dpi = 100)
    plt.close()
    print "\n2) {0} plot saved.".format(figname)
    answer_description_3_2()

    # part 3 - examine partial dependence of two features on target
    # note 'char_freq_1' is has index = 51, 'word_freq_remove' = 6
    # when this solution was written
    plot_partial_dependence(model, X_train, [(51, 6)],
                            feature_names = all_feature_names,
                            figsize=(10,10))
    plt.tight_layout()
    plt.legend()
    figname = '3_3_partial_dependency_plots.png'
    plt.savefig(figname, dpi = 100)
    plt.close()
    print "\n3) {0} plot saved.".format(figname)

    # part 4 - 3D surface showing partial dependence of two features on target
    pdp_3d(model, (51, 6), X_train, all_feature_names)



if __name__ == '__main__':
    df = pd.read_csv('data/churn_train.csv')
    df = df.fillna(df.mean())
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['active'] = df['last_trip_date'].dt.month >= 6
    df = pd.get_dummies(df, columns=['city', 'phone'])
    # print df.head()

    df.drop(['last_trip_date', 'signup_date'], axis=1, inplace=True)


    y = df.pop('active').values
    X = df.values
    X_scaled = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,y)

    SVC_tester(X_train, y_train, X_test)

    #Use Boosting pair.py code to grid search and plot boosting:
    model_best = part_2(X_train, X_test, y_train, y_test)


    #John's tweeked model for boosting (used gridsearch to set hyperparams)
    est = GradientBoostingClassifier(n_estimators=1000,
                                    learning_rate= 0.1,
                                    max_depth=3,
                                    max_features='sqrt')
    model = est.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = model.score(X_test, y_test)

    sort_feat = np.argsort(model.feature_importances_)[::-1]
    rel_feat_imp = model.feature_importances_/max(model.feature_importances_)

    py_ind = np.arange(12, -1, -1) # 12 to 0
    fig = plt.figure(figsize=(8, 8))
    plt.barh(y_ind, rel_feat_imp[sort_feat], height = 0.6, align='center',
        color = 'b', alpha=0.7)
    plt.ylim(y_ind.min() + 0.5, y_ind.max() + 0.5)
    plt.yticks(y_ind, feature_names[sort_feat])
    plt.xlabel('Relative feature importances')
    plt.ylabel('Features');

    #Partial dependency plots:
    partial_dependence.plot_partial_dependence(model, X_train, sort_feat,
                            feature_names = feature_names[sort_feat],
                            figsize=(10,10))
    plt.tight_layout();
