"""
 This package is heavily influenced by Multiple Imputation(MI)  algorith proposed in
 Multiple Imputation Through XGBoost, Yongshi Deng, Thomas Lumley , https://doi.org/10.48550/arXiv.2106.01574

 The authers proposes a novel scalable MI framework mixgb, which is based on XGBoost, subsampling, and
 Predictive Mean Matching (PMM). The authers' R implementation of algorithm is available at https://github.com/agnesdeng/mixgb
 and on CRAN. The R package has implemented three types of PMM, but I have only implemented
 the type 2, which seems to give the best results in most of the scenarios.

 I have extended the proposed algorithm to create trees based on Random Forrest, Catboost, GradientBoost
 or HistGradientBoost. TrIMe gives the option to use the default parameters for tree ensembles, or you
 can pass custom parameters. There is also an option to do an initial hyperparameter optimization and
 use the optimized parameters for all subsequent training. To run the initial hyperparameter optimization,
 data is imputed using the initial_num/int/cat values, and optuna is used to find the optimal parameter values


 Another change I have made is at the initial imputation level. For positive continuous variable, it does a
 log transformation before doing a normal sampling.

 The name of the package, TrIMe is a playful anagram of TREe Multiple Imputation.
 The API of the library is consistent with the scikit-learn
 
"""



import pandas as pd
import numpy as np

try:
    import xgboost as xgb
except:
    xgb = None

try:
    import catboost as cat
except:
    cat = None

import optuna
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split
from sklearn.metrics import get_scorer

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute._base import _BaseImputer

import utils

class TriMe(_BaseImputer):

    def __init__(self, model_type = 'xgb',
                 gpu_id = None,
                 opt = False,
                 opt_trials = 30,
                 kfold = 5,
                 max_iter = 1,
                 max_trees = None,
                 num_datasets = 5,
                 pmm = True,
                 pmm_k = 5,
                 target_var = None,
                 ordinalAsInt = False,
                 pmm_link = 'logitraw',
                 initial_num="normal",
                 initial_int="mode",
                 initial_cat="sample",
                 override = {},
                 verbose = 0,
                 random_state = 42,
                 tree_params={}):

        """
        Multiple Imputer using various tree based models for imputing missing values.

        Parameters
              model_type     - (default xgb), xgb  - xgboost
                                              cat  - catboost
                                              gbt  - gradient boosting
                                              hgbt - histogram based gradient boosting
                                              rf   - random forest
              gpu_id         - (default None) pass gpu_id to use GPU. GPU is only supported for catboost and xgboost.
              num_datasets   - (default 5) The number of imputed datasets.
              max_iter       - (default 1) The number of imputation iterations.
              pmm            - (default True)  predictive mean matching
              pmm_k          - (default 5) The number of donors for pmm
              initial_num    - (default normal) Initial imputation method for numeric type data ("normal","mean","median","mode","sample").
              initial_int    - (default mode) Initial imputation method for integer type data ("mode","sample").
              initial_cat    - (default sample) Initial imputation method for categorical type data ("mode","sample").
              opt            - (default False) option to deplot initial hyperparameter optimization
              opt_trials     - (default 20) number of optuna trials for initial hyper parameter optimization
              max_trees      - (default None ) this overrides the default number of boosted trees if used along with opt = False
                               When used with opt=True this is the max number of boosted trees used in cross validation with early stopping.
                               When passing custome parametrs using tree_params, the number of estimators get updated to max_trees
              override       - is a dictionary of column_name : method. It overrides the general methods defined by initial_num, initial_int and initial_cat
              ordinalAsInt   - Whether to convert ordinal factors to integers. ordinalAsInt = TRUE may speed up the imputation process for large datasets.
              pmm_link       - (default logitraw), The link function for predictive mean matching in binary variables. (logitraw, logit)
              kfold          - Number of folds used in cross validation during the hyperparameter optimization
              verbose        - (default 0) 1 for verbosity
              tree_params    - custum parameter name-value pairs to use when opt =False, otherwise default parameter values are used

        Methods:
            fit_transform(X):
                Fits the imputation models on the provided data `X` and returns the imputed datasets.

            fit(X):
                Fits the imputation models on the provided data `X`.

            transform(X):
                Transforms the provided data `X` using the fitted imputation models and returns the imputed datasets.

        """

        _allowed_types = ['xgb', 'cat', 'gbt', 'hgbt', 'rf']
        if xgb is None:
            _allowed_types.remove('xgb')

        if cat is None:
            _allowed_types.remove('cat')

        assert model_type in _allowed_types, f"Tree type {model_type} not supported, or package not installed "


        # set random seed
        np.random.seed(random_state)

        self.model_type = model_type
        self.initial_num = initial_num
        self.initial_int = initial_int
        self.initial_cat = initial_cat
        self.override = override
        self.ordinalAsInt = ordinalAsInt
        self.pmm_link = pmm_link
        self.pmm = pmm
        self.pmm_k = pmm_k
        self.num_datasets = num_datasets
        self.max_iter=max_iter
        self.max_trees = max_trees
        self.initial_impute = None
        self.gpu_id = gpu_id
        self.opt = opt
        self.opt_trials = opt_trials
        self.kfold = kfold
        self.tree_params = tree_params
        self.target_var = target_var
        self.verbose = verbose
        self.random_state = random_state
        # estimated models and parameters
        self.opt_tree_params = {}
        # clean the categorical data using LabelEncoder
        # xgboost needs categories in range[0,num_class), but data may have some other range of categories.
        # LabelEncoder fixes this specific problem
        self._encoders = {}


        if self.max_trees is not None:
            if self.model_type == 'cat':
                self.tree_params.update({'iterations' : self.max_trees})
            elif self.model_type == 'hgbt':
                self.tree_params.update({'max_iter': self.max_trees})
            else:
                self.tree_params.update({'n_estimators': self.max_trees})

        # optuna uses various subsampling paramters available with each model
        # to match with  Yongshi Deng, et.el.  add default subsample=0.7  self.opt == False
        if not self.opt:
            if self.model_type in ( 'xgb','cat','gbt'):
                d = {'subsample':0.7}
                d.update(self.tree_params)
                self.tree_params = d
            elif self.model_type == 'rf':
                d = {'max_samples': 0.7}
                d.update(self.tree_params)
                self.tree_params = d





        # train_dir for catboost
        train_dir = None
        if os.name == 'nt':
            train_dir = r'C:\Users\xxxxx\AppData\Local\Temp'.replace('xxxxx', os.environ['USERNAME'])
        elif os.name == 'posix':
            train_dir = os.environ.get('TMPDIR') or '/tmp'

        self.train_dir = train_dir

        self.initial_impute = None
        # all models. Default behavior is to train all models
        self._models = {}

    def _data_type(self, x):
        """
            Default evaluation metrics for different method_types and data types
            Parameters:
                x - Series type

        """

        if pd.api.types.is_categorical_dtype(x):
            if len(x.cat.categories) == 2:
                d_type = 'binary'
            else:
                d_type = 'multiclass'
        else:
            d_type = 'number'

        return d_type


    def _eval_metric(self, d_type):
        """
            Default evaluation metrics for different method_types and data types
            Parameters:
                d_type - varaible type -- number(int,float), binary or multiclass variables

        """

        if d_type == 'number':
            eval_metric = 'rmse'
            # note that sklearn based models cannot handle rmse.
            # for these models, the eval_metric is changed to neg_mean_squared_error
            # during the evaluation process, and converted to rmse later
        else:
            if self.model_type == 'xgb':
                # decide the eval_metric based on target_new type
                if d_type == 'binary':
                    eval_metric = 'logloss'
                else:
                    eval_metric = 'mlogloss'

            elif self.model_type == 'cat':

                if d_type == 'binary':
                    eval_metric = 'Logloss'
                else:
                    eval_metric = 'MultiClass'

            else:
                if d_type == 'binary':
                    eval_metric = 'roc_auc'
                else:
                    eval_metric = 'roc_auc_ovr'

        return eval_metric


    def _max_boosted_trees(self):
        """
            Default parameter for max iterations/trees and early stopping

        """

        if self.model_type in ('xgb', 'cat'):
            early_stopping_rounds =  20
        elif self.model_type in ('gbt', 'hgbt'):
            early_stopping_rounds =  10
        elif self.model_type == 'rf':
            early_stopping_rounds =  5

        if self.max_trees is not None:
            max_trees =  self.max_trees

        else:
            if self.model_type in ('xgb','cat'):
                max_trees =  5000
            elif self.model_type in ('gbt','hgbt'):
                max_trees =  2000
            elif self.model_type == 'rf':
                max_trees =  500

        return max_trees, early_stopping_rounds

    def _custom_cross_val_score(self,estimator, X, y,  kf=KFold(n_splits=5),  scoring=None):
        # this solution is inspired by https://github.com/microsoft/LightGBM/pull/3204

        best_iterations = []

        if scoring == 'rmse':
            scorer = get_scorer('neg_mean_squared_error')
        else:
            scorer = get_scorer(scoring)


        is_histboost = 'Hist' in str(estimator.__str__)
        is_randomforest = 'Random' in str(estimator.__str__)

        # random forest doesn't have early stopping
        if not is_randomforest:
            for train_index, test_index in kf.split(X,y):
                X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

                estimator.fit(X_train, y_train)
                if is_histboost:
                    best_iteration = estimator.n_features_in_
                else:
                    best_iteration = estimator.n_estimators_


                best_iterations.append(best_iteration)

            average_best_iteration = int(np.mean(best_iterations))
            estimator.n_estimators = average_best_iteration

            if is_histboost:
                estimator.n_iter_no_change = 1
            else:
                estimator.n_iter_no_change = None

        else:
            average_best_iteration = estimator.n_estimators

        final_scores = []
        for train_index, test_index in kf.split(X,y):
            X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

            estimator.fit(X_train, y_train)

            score = scorer(estimator,X_test, y_test)

            final_scores.append(score)

        if scoring == 'rmse':
            final_scores = [ np.sqrt(-x) for x in final_scores]

        return np.array(final_scores), average_best_iteration

    def _opt_hyper(self, X, cat_columns_v):
        """
            Finds optimal paramters for self.model_type

            Parameters:
                X (pandas.DataFrame): Input features.
                cat_columns_v (list): List of categorical columns.

        """


        # make a copy to avoid making any changes
        data = X.copy()
        cat_columns = cat_columns_v.copy()

        if self.verbose:
            print(f'Running initial hyperparameter optimization using target var = {self.target_var}')


        target = data.pop(self.target_var)
        target_type = self._data_type( target)

        if target_type == 'number':
            kfolds = KFold(n_splits=self.kfold)
        else:
            kfolds = StratifiedKFold(n_splits=self.kfold)

        # decide the eval_metric based on target_new type
        scoring = self._eval_metric(target_type)

        # remove target_new from cat_columns if it is a categorical type
        if self.target_var in cat_columns:
            cat_columns.remove(self.target_var)

        max_trees,early_stopping_rounds = self._max_boosted_trees()

        if self.model_type == 'xgb':

            def objective_xgb(trial):

                xgb_params = dict(
                    max_depth=trial.suggest_int("max_depth", 2, 10),
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                    subsample=trial.suggest_float("subsample", 0.6, 0.9),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True)
                )

                if self.gpu_id is not None:
                    xgb_params['tree_method'] = 'gpu_hist'
                    xgb_params['gpu_id'] = self.gpu_id

                cv_results = xgb.cv(params=xgb_params,
                    dtrain=xgb.DMatrix(data, label=target, enable_categorical = True),
                    early_stopping_rounds=early_stopping_rounds,
                    folds=kfolds,
                    metrics = scoring,
                    num_boost_round=max_trees,
                    seed=self.random_state)

                score = cv_results['test-' + scoring + '-mean'].values[-1]


                return score

            study = optuna.create_study(direction="minimize")
            study.optimize(objective_xgb, n_trials=self.opt_trials)

            self.opt_tree_params = study.best_params
            self.opt_tree_params['n_estimators'] = max_trees
            self.opt_tree_params['early_stopping_rounds'] = early_stopping_rounds


            if self.gpu_id is not None:
                self.opt_tree_params['tree_method'] = 'gpu_hist'
                self.opt_tree_params['gpu_id'] = self.gpu_id


        elif self.model_type == 'cat':

            def objective_cat(trial):

                cat_params = dict(early_stopping_rounds=early_stopping_rounds,
                                  loss_function=scoring,
                                  train_dir = self.train_dir,
                                  num_boost_round=max_trees,
                                  depth=trial.suggest_int("depth", 2, 10),
                                  learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                                  min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 10),
                                  colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.5, 1.0),
                                  subsample=trial.suggest_float("subsample", 0.6, 0.9),
                                  l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-4, 1e2, log=True),
                                  random_strength=trial.suggest_float("random_strength", .2, 1, log=True))


                if self.gpu_id is not None:
                    # colsample_bylevel is not implemented for GPU
                    # bootstrap_type has to be Bernoulli to run on GPU
                    cat_params['task_type'] = 'GPU'
                    cat_params['devices'] = str(self.gpu_id)
                    cat_params['bootstrap_type']='Bernoulli'
                    cat_params.pop('colsample_bylevel')

                cat_params['silent'] = True
                cat_params['cat_features'] = cat_columns

                cv_results = cat.cv(params=cat_params,
                                    pool=cat.Pool(data = data, label = target, cat_features=cat_columns),
                                    folds=kfolds,
                                    seed=self.random_state,
                                    return_models=True)

                score = cv_results[0]['test-' + scoring + '-mean'].values[-1]


                return score


            study = optuna.create_study(direction="minimize")

            study.optimize(objective_cat, n_trials=self.opt_trials)
            self.opt_tree_params = study.best_params
            self.opt_tree_params['iterations'] = max_trees
            self.opt_tree_params['early_stopping_rounds'] = early_stopping_rounds
            self.opt_tree_params['train_dir'] = train_dir


            if self.gpu_id is not None:
                # colsample_bylevel is not implemented for GPU
                # bootstrap_type has to be Bernoulli to run on GPU
                self.opt_tree_params.pop('colsample_bylevel')
                self.opt_tree_params['task_type'] = 'GPU'
                self.opt_tree_params['devices'] = str(self.gpu_id)
                self.opt_tree_params['bootstrap_type'] = 'Bernoulli'
                self.opt_tree_params['silent'] = True

        elif self.model_type == 'gbt':

            def objective_gbt(trial):

                # Define the search space for hyperparameters
                gb_params = dict(n_estimators=max_trees,
                                 validation_fraction=0.2,
                                 n_iter_no_change=early_stopping_rounds,
                                 subsample=trial.suggest_float("subsample", 0.6, 0.9),
                                 max_features=trial.suggest_float("max_features", .5, 1),
                                 ccp_alpha=trial.suggest_float("ccp_alpha", 1e-4, 1e2, log=True),
                                 learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                                 max_depth=trial.suggest_int("max_depth", 2, 9))

                # Create a Random Forest classifier with the sampled hyperparameters
                if target_type == 'number':
                    gb_model = GradientBoostingRegressor(**gb_params)
                else:
                    gb_model = GradientBoostingClassifier(**gb_params)

                score, n_estimators = self._custom_cross_val_score(gb_model, data, target, kf=kfolds, scoring=scoring)

                return score.mean()

            if scoring in ('roc_auc', 'roc_auc_ovr'):
                study = optuna.create_study(direction="maximize")
            else:
                study = optuna.create_study(direction="minimize")

            study.optimize(objective_gbt, n_trials=self.opt_trials)

            self.opt_tree_params = study.best_params

            self.opt_tree_params['n_estimators'] = max_trees
            self.opt_tree_params['n_iter_no_change'] = early_stopping_rounds
            self.opt_tree_params['validation_fraction'] = 0.2




        elif self.model_type == 'hgbt':

            def objective_hgbt(trial):

                # Define the search space for hyperparameters
                hg_params = dict (max_iter=max_trees,
                                   validation_fraction=0.2,
                                    n_iter_no_change=early_stopping_rounds,
                                        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                                        l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 1e2, log=True),
                                            max_depth = trial.suggest_int("max_depth",2, 9))

                # Create a Random Forest classifier with the sampled hyperparameters
                if target_type == 'number':
                    if self.ordinalAsInt:
                        hg_model = HistGradientBoostingRegressor(**hg_params)
                    else:
                        hg_model = HistGradientBoostingRegressor(categorical_features=data.columns.isin(cat_columns),
                                                                 **hg_params)
                else:
                    if self.ordinalAsInt:
                        hg_model = HistGradientBoostingClassifier(**hg_params)
                    else:
                        hg_model = HistGradientBoostingClassifier(categorical_features=data.columns.isin(cat_columns),
                                                              **hg_params)


                score, n_estimators = self._custom_cross_val_score(hg_model, data, target, kf=kfolds, scoring=scoring)
                trial.set_user_attr('num_iters', n_estimators)

                return score.mean(),

            if scoring in ('roc_auc', 'roc_auc_ovr'):
                study = optuna.create_study(direction="maximize")
            else:
                study = optuna.create_study(direction="minimize")
            study.optimize(objective_hgbt, n_trials=self.opt_trials)
            self.opt_tree_params = study.best_params

            self.opt_tree_params['max_iter'] = max_trees
            self.opt_tree_params['n_iter_no_change'] = early_stopping_rounds
            self.opt_tree_params['validation_fraction'] = 0.2


        elif self.model_type == 'rf':

            def objective_rf(trial):
                # Define the search space for hyperparameters

                rf_params = dict(ccp_alpha=trial.suggest_float("ccp_alpha", 1e-4, 1e2, log=True),
                                n_estimators = trial.suggest_int("n_estimators", 50, 400, step=10),
                                max_depth = trial.suggest_int("max_depth", 2, 7),
                                max_samples=trial.suggest_float("max_samples", 0.6, 0.9),
                                min_samples_split = trial.suggest_int("min_samples_split", 2, 20))

                # Create a Random Forest classifier with the sampled hyperparameters
                if target_type == 'number':
                    rf_model = RandomForestRegressor(**rf_params)
                else:
                    rf_model = RandomForestClassifier(**rf_params)

                # decide the eval_metric based on target type
                scoring = self._eval_metric(target_type)

                score, n_estimators = self._custom_cross_val_score(rf_model, data, target, kf=kfolds,scoring=scoring)


                return score.mean()

            if scoring in ('roc_auc', 'roc_auc_ovr'):
                study = optuna.create_study(direction="maximize")
            else:
                study = optuna.create_study(direction="minimize")
            study.optimize(objective_rf, n_trials=self.opt_trials)
            self.opt_tree_params = study.best_params


    def _fit_models(self, X, y, cat_columns):

        """
            Fits the models based on the specified model_type and paramaters(tree_params or opitmized params created by self_opt_hyper) for imputing missing values.

            Parameters:
                X (pandas.DataFrame): Input features.
                y (pandas.Series): Target variable.
                cat_columns (list): List of categorical columns.

            Returns:
                model: Fitted imputation model.
        """


        # get the user supplied tree params
        tree_params = self.tree_params
        # and update with optimized parameters
        if tree_params is not None:
            tree_params.update(self.opt_tree_params)
        else:
            tree_params = self.opt_tree_params


        y_type = self._data_type(y)

        if self.model_type == 'xgb':

            if y_type == 'number':
                model = xgb.XGBRegressor(enable_categorical=True,**tree_params)
            else:
                model = xgb.XGBClassifier(enable_categorical=True,**tree_params)

            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],verbose=False)

        elif self.model_type == 'cat':

            if len(cat_columns) > 0:
                tree_params['cat_features'] = cat_columns

            if y_type == 'number':
                model = cat.CatBoostRegressor(**tree_params)
            else:
                model = cat.CatBoostClassifier(**tree_params)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


            model.fit(X_train, y_train, eval_set=[ (X_test, y_test)], verbose=False)

        elif self.model_type == 'gbt':

            # Create a Random Forest classifier with the sampled hyperparameters
            if y_type == 'number':
                model = GradientBoostingRegressor(**tree_params)
            else:
                model = GradientBoostingClassifier(**tree_params)

            model.fit(X, y)

        elif self.model_type == 'hgbt':

            if y_type == 'number':
                if self.ordinalAsInt:
                    model = HistGradientBoostingRegressor(**tree_params)
                else:
                    model = HistGradientBoostingRegressor(categorical_features=X.columns.isin(cat_columns),
                                                             **tree_params)
            else:
                if self.ordinalAsInt:
                    model = HistGradientBoostingClassifier(**tree_params)
                else:
                    model = HistGradientBoostingClassifier(categorical_features=X.columns.isin(cat_columns),
                                                              **tree_params)

            model.fit(X, y)

        elif self.model_type == 'rf':

            # Create a Random Forest classifier with the sampled hyperparameters
            if y_type == 'number':
                model = RandomForestRegressor(**tree_params)
            else:
                model = RandomForestClassifier(**tree_params)


            model.fit(X, y)


        return model

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)



    def fit(self, X):
        """
            Fits the imputation models on the provided data `X`.

            Parameters:
                X (pandas.DataFrame): Input features.
        """

        self.initial_impute = utils.InitialImputer(initial_num = self.initial_num, initial_int = self.initial_int, initial_cat = self.initial_cat, override=self.override, verbose = self.verbose)
        X = self.initial_impute.fit_transform(X)

        # check if the ordinal variables are to be converted to integer
        cat_columns = list(X.select_dtypes(include='category').columns)
        if self.ordinalAsInt:
            X[cat_columns] = X[cat_columns].astype("Int64")
        else:
            # use labelencoder to clean the categorical data
            # some methods do not work if some categories are missing e.g. xgboost does not work
            # if data has [-2,-1,1,2] categories. Labelencoder encodes categories in the range [0,# category]
            for col in cat_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                X[col] = X[col].astype('category')
                self._encoders[col] = le


        # run initial hyperparameter optimization
        # self._opt_hyper gets cross-validated best parametrs
        # the optimized params are used for all the models trained later on
        # this step could be slow, but this only needs to be done once
        if self.opt:
            self._opt_hyper(X, cat_columns)


        for iter in range(1, self.max_iter+1):
            cols = X.columns
            for col_ind in range(len(cols)):

                var = cols[col_ind]

                if self.verbose:
                    print(f"Iteration {iter} for {var}")

                # get nan index for column var
                na_idx = self.initial_impute.na_idx[var]
                obs_idx = self.initial_impute.obs_idx[var]
                target_data = X[var]
                data = X.drop(var, axis = 1)
                var_type = self._data_type(target_data)

                yobs = target_data.iloc[obs_idx]
                c_c = cat_columns.copy()
                if var in cat_columns:
                    c_c.remove(var)

                var_model = self._fit_models(data.iloc[obs_idx,:], yobs, c_c)


                # For multiple iterations (max_iter > 1) only the last model is saved
                self._models[var] = var_model

                # iterate over self.max_iter times
                if iter < self.max_iter:

                    if var_type == 'number':

                        yhatmis = var_model.predict(data.iloc[na_idx,:])
                        yhatobs = var_model.predict(data.iloc[obs_idx,:])
                        yhat = utils.pmm_numeric_binary(yhatobs, yhatmis, yobs, self.pmm_k)

                    else:

                        if var_type == 'binary':

                            yhatmis = var_model.predict_proba(data.iloc[na_idx, :])[:, 1]
                            yhatobs = var_model.predict_proba(data.iloc[obs_idx, :])[:, 1]

                            if self.pmm_link == 'logitraw':
                                yhatmis = np.vectorize(lambda p : np.log(p/(1-p)))(yhatmis)
                                yhatobs = np.vectorize(lambda p : np.log(p/(1-p)))(yhatobs)
                            else:
                                pass

                            yhat = utils.pmm_numeric_binary(yhatobs, yhatmis, yobs, self.pmm_k)

                        else:

                            yhatmis = var_model.predict_proba(data.iloc[na_idx, :])
                            yhatobs = var_model.predict_proba(data.iloc[obs_idx, :])

                            yhat = utils.pmm_multiclass(yhatobs, yhatmis, yobs, self.pmm_k)


                    X.iloc[na_idx, col_ind] = yhat


        return self

    def transform(self, X):
        """
            Transforms the provided data `X` using the fitted imputation models and returns the imputed datasets.

            Parameters:
                X (pandas.DataFrame): Input features.

            Returns:
                list: List of num_datasets imputed datasets.

        """

        # self.params has the
        if self.initial_impute is None:
            raise ValueError("fit() method must be called before transform().")

        # do the initial imputation and basic transforms
        # initial imputation
        X = self.initial_impute.transform(X)
        # check if the ordinal variables are to be converted to integer
        cat_columns = X.select_dtypes(include='category').columns
        if self.ordinalAsInt:
            X[cat_columns] = X[cat_columns].astype("Int64")
        else:
            # use labelencoder to clean the categorical data
            for col in cat_columns:
                le = self._encoders[col]
                X[col] = le.fit_transform(X[col])


        imputed_datasets = []

        for i in range(self.num_datasets):
            x_data = X.copy()
            cols = X.columns
            for col_ind in range(len(cols)):

                var = cols[col_ind]

                # get nan index for column var
                na_idx = self.initial_impute.na_idx[var]
                obs_idx = self.initial_impute.obs_idx[var]
                target_data = x_data[var]
                data = X.drop(var, axis=1)
                var_type = self._data_type(target_data)

                yobs = target_data.iloc[obs_idx]
                var_model = self._models[var]

                if var_type == 'number':

                    yhatmis = var_model.predict(data.iloc[na_idx, :])
                    yhatobs = var_model.predict(data.iloc[obs_idx, :])
                    yhat = utils.pmm_numeric_binary(yhatobs, yhatmis, yobs, self.pmm_k)

                else:

                    if var_type == 'binary':

                        print(var)
                        yhatmis = var_model.predict_proba(data.iloc[na_idx, :])[:, 1]
                        yhatobs = var_model.predict_proba(data.iloc[obs_idx, :])[:, 1]

                        if self.pmm_link == 'logitraw':
                            yhatmis = np.vectorize(lambda p: np.log(p / (1 - p)))(yhatmis)
                            yhatobs = np.vectorize(lambda p: np.log(p / (1 - p)))(yhatobs)
                        else:
                            pass

                        yhat = utils.pmm_numeric_binary(yhatobs, yhatmis, yobs, self.pmm_k)

                    else:

                        yhatmis = var_model.predict_proba(data.iloc[na_idx, :])
                        yhatobs = var_model.predict_proba(data.iloc[obs_idx, :])

                        yhat = utils.pmm_multiclass(yhatobs, yhatmis, yobs, self.pmm_k)

                if (not self.ordinalAsInt) and (var in cat_columns):
                    # map labelencoder to original data
                    le = self._encoders[col]
                    X[col] = le.transform(X[col])

                X.iloc[na_idx, col_ind] = yhat


            imputed_datasets.append( x_data)

        return imputed_datasets


