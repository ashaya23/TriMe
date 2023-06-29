"""
 utility functions and classes for TrIme package
 
"""

from sklearn.impute._base import _BaseImputer
from sklearn.neighbors import NearestNeighbors

import pandas as pd
import numpy as np
from pandas.api.types import  is_integer
import warnings

class InitialImputer(_BaseImputer):

    def __init__(self, initial_num="normal", initial_int="mode", initial_cat="mode", override = {}, verbose = 0):
        #  initial_num  - Initial imputation method for numeric type data ("normal","mean","median","mode","sample"). Default: "normal"
        #  initial_int  - Initial imputation method for integer type data ("mode","sample"). Default: "mode"
        #  initial_cat  - Initial imputation method for categorical type data ("mode","sample"). Default: "mode"
        #  override     - is a dictionary of column_name : method. It overrides the general methods defined by initial_num, initial_int and initial_cat
        #  verbose      - (default 0) 1 for verbosity
        self.initial_num = initial_num
        self.initial_int = initial_int
        self.initial_cat = initial_cat
        self.override = override
        self.verbose = verbose

        self.sorted_dt = None
        self.origin_names = None
        self.sorted_idx = None
        self.sorted_col_names = None
        self.sorted_types = None
        self.sorted_na_sums = None
        self.origin_names = None
        self.n_row = None
        self.n_col = None
        self.missing_num = None
        self.missing_vars = None
        self.missing_types = None
        self.missing_method = None
        self.obs_idx = None
        self.na_idx = None
        self.params = {}

    def fit(self, data):

        X = data.copy()

        # erase any previous values, if populated
        self.params = {}

        n_row, n_col = X.shape

        if n_col < 2:
            raise ValueError("Data needs to have at least two columns.")


        # Data preprocessing
        # 1) Sort the dataset with increasing NAs
        origin_names = X.columns
        sorted_dt, sorted_idx, sorted_col_names = self._sortNA(X)
        sorted_types = self._feature_type(X)

        # 2) Initial imputation & data validation
        sorted_na_sums = sorted_dt.isna().sum()
        missing_cols = sorted_na_sums[sorted_na_sums > 0].index
        missing_types = sorted_types[missing_cols].to_list()

        missing_method = pd.Series([
            self.initial_num if var_type == np.number else
            (self.initial_int if (var_type == np.int) or is_integer(var_type) else self.initial_cat)
            for var_type in missing_types], index=missing_cols)

        # update with self.override
        missing_method.update(pd.Series(self.override, dtype = 'str'))


        if all(sorted_na_sums == 0):
            raise ValueError("No missing values in this DataFrame.")

        if any(sorted_na_sums == n_row):
            raise ValueError("At least one variable in the DataFrame has 100% missing values.")

        if any(sorted_na_sums >= 0.9 * n_row):
            print("Warning: Some variables have more than 90% missing entries.")

        missing_num = len(missing_cols)
        obs_idx = {}
        na_idx = {}

        for var in missing_cols:
            na_idx[var] = np.where(sorted_dt[var].isna())[0]
            obs_idx[var] = np.where(~sorted_dt[var].isna())[0]

            if missing_method[var] == "normal":
                self._imp_normal(sorted_dt[var], na_idx[var], impute= False)
            elif missing_method[var] == "mean":
                self._imp_mean(sorted_dt[var], na_idx[var], impute= False)
            elif missing_method[var] == "median":
                self._imp_median(sorted_dt[var], na_idx[var], impute= False)
            elif missing_method[var] == "mode":
                self._imp_mode(sorted_dt[var], na_idx[var], impute= False)
            elif missing_method[var] == "sample":
                self._imp_sample(sorted_dt[var], na_idx[var], impute= False)
            else:
                raise ValueError(f"{missing_method[var]} imputation method not recognized")


        self.sorted_col_names = sorted_col_names
        self.origin_names = origin_names
        self.sorted_types = sorted_types
        self.sorted_na_sums = sorted_na_sums
        self.origin_names = X.columns.tolist()
        self.n_row = n_row
        self.n_col = n_col
        self.missing_num = missing_num
        self.missing_vars = missing_cols
        self.missing_types = missing_types
        self.missing_method = missing_method
        self.obs_idx = obs_idx
        self.na_idx = na_idx


        return self

    def transform(self, X):
        # self.params has the
        if len(self.params) == 0:
            raise ValueError("fit() method must be called before transform().")

        # Perform imputation on X
        transformed_dt = X.copy()

        for var in self.missing_vars:
            na_idx = np.where(transformed_dt[var].isna())[0]
            obs_idx = np.where(~transformed_dt[var].isna())[0]

            if self.missing_method[var] == "normal":
                transformed_dt[var] = self._imp_normal(transformed_dt[var], na_idx, impute= True)
            elif self.missing_method[var] == "mean":
                transformed_dt[var] = self._imp_mean(transformed_dt[var], na_idx, impute= True)
            elif self.missing_method[var] == "median":
                transformed_dt[var] = self._imp_median(transformed_dt[var], na_idx, impute= True)
            elif self.missing_method[var] == "mode":
                transformed_dt[var] = self._imp_mode(transformed_dt[var], na_idx, impute= True)
            elif self.missing_method[var] == "sample":
                transformed_dt[var] = self._imp_sample(transformed_dt[var], na_idx, impute= True)

            # na_idx and obs_idx stores the missing and non-missing values of tradnsformed data
            # na_idx and obs_idx values for training data is not relevant at this point
            self.obs_idx[var] = obs_idx
            self.na_idx[var] = na_idx

        # return in increasing order of NaN values
        return transformed_dt.reindex(self.sorted_col_names, axis = 1)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


    def _feature_type(self,data):


        types = data.dtypes

        if 'object' in types.values:
            raise ValueError("Data contains variables of object type. Please change them into categorical.")

        ord_fac = []
        for col in data.select_dtypes(include=['category']):
            if isinstance(data[col].dtype, pd.CategoricalDtype) and data[col].dtype.ordered:
                ord_fac.append(col)

        if len(ord_fac) > 0:
            types.loc[ord_fac] = 'category'

        factor_vars = types[types == 'category'].index
        # this info is used to decide what loss function to use for boosting algorithm
        # e.g. squarederror for continuous, while softmax for multiclass
        for var in factor_vars:
            levels = data[var].cat.categories
            if len(levels) == 2:
                types[var] = 'binary'
            else:
                types[var] = 'multiclass'

        return types


    def _sortNA(self,data):
        # sort data columns in increasing order of na values.
        # column with the smallest number of NaN arranged first, and with most is arrranged last

        sorted_data = data[data.isna().sum().sort_values().index]
        sorted_col_names = sorted_data.columns.tolist()
        sorted_idx = data.columns.get_indexer_for(sorted_col_names)

        return sorted_data,  sorted_idx,  sorted_col_names



    def _imp_sample(self,vec, na_idx = None, impute=False):
        # used for categorical variables.
        # sampling with replacement ensures that the class distribution of
        # imputed rows is close to the distribution of available data

        name = vec.name
        vec = vec.values

        prob_dist = self.params.get(name, pd.Series(dtype='float64'))

        if  len(na_idx) == 0:
            na_idx = np.where(pd.isna(vec))[0]

        n_na = len(na_idx)

        if prob_dist.empty:
            prob_dist = vec[~na_idx].value_counts()
            prob_dist = prob_dist / prob_dist.sum()
            self.params[name] = prob_dist

        if impute:
            vec[na_idx] = np.random.choice(prob_dist.index.to_numpy(), size=n_na, p=prob_dist.to_numpy(dtype='float64'))

        return vec


    def _imp_normal(self,vec, na_idx = [], impute=False):
        # the origianl R code has a problem with continuous variables which are all positive
        # e.g. some variables like age and height are always positive.
        # This function checks if all values are positive does a log transformation
        # in case they are

        name = vec.name
        vec = vec.values

        if not np.issubdtype(vec.dtype, np.number):
            raise ValueError(f"{name} : _imp_normal(vec, ...) only applies to a numeric vector.")

        if len(na_idx) == 0:
            na_idx = np.where(np.isnan(vec))[0]
        n_na = len(na_idx)
        obs_idx = np.where(~np.isnan(vec))[0]

        log_transform = False
        # make an initial guess about log_transform
        # this will get over written if the training data value is saved
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not any(vec < 0):
                log_transform = True


        # check if the model is already trained
        train_params = self.params.get(name, {})
        # note that the log_transform is overwritten here based on the value from training data
        log_transform = train_params.get('log_transform',log_transform)

        if log_transform:
            if  (not impute):
                if self.verbose:
                    print(f"{name} : all values are positive. Applying log transformation")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vec = np.log1p(vec)
            orig_vec = vec[obs_idx]


        if train_params:
            var_mean = train_params['mean']
            var_sd = train_params['std']
        else:
            var_mean = np.nanmean(vec)
            var_sd = np.nanstd(vec)

        self.params[name] = {'log_transform' : log_transform,  'mean' :  var_mean, 'std' :  var_sd}


        if impute:
            vec[na_idx] = np.random.normal(loc=var_mean, scale=var_sd, size=n_na)

            # change back to the original scale
            if log_transform:
                vec = np.expm1(vec)
                # use the orig_vec for non-nan values. This deals with any sort of rounding issues
                # this is probably unneccesary in most of the cases.
                vec[obs_idx] = orig_vec

        return vec


    def _imp_mean(self,vec, na_idx = [], impute=False):

        name = vec.name
        vec = vec.values

        if not np.issubdtype(vec.dtype, np.number):
            raise ValueError(f"{name} : _imp_mean(vec, ...) only applies to a numeric vector.")

        if  len(na_idx) == 0:
            na_idx = np.where(np.isnan(vec))[0]

        train_params = self.params.get(name, False)

        if  train_params:
            var_mean = train_params['mean']
        else:
            var_mean = np.nanmean(vec)
            self.params[name] = {'mean': var_mean}


        if impute:
            vec[na_idx] = np.repeat(var_mean, len(na_idx))

        return vec

    def _imp_median(self, vec, na_idx = [], impute=False):

        name = vec.name
        vec = vec.values

        if not np.issubdtype(vec.dtype, np.number):
            raise ValueError(f"{name} : imp_median(vec, ...) only applies to a numeric vector.")

        if  len(na_idx) == 0:
            na_idx = np.where(np.isnan(vec))[0]

        train_params = self.params.get(name, False)

        if  train_params:
            var_median = train_params['median']
        else:
            var_median = np.nanmedian(vec)
            self.params[name] = {'median': var_median}

        if impute:
            vec[na_idx] = np.repeat(var_median, len(na_idx))

        return vec


    def _imp_mode(self, vec, na_idx = [], impute=False):

        name = vec.name
        vec = vec.values


        if  len(na_idx) == 0:
            na_idx = np.where(np.isnan(vec))[0]

        train_params = self.params.get(name, False)

        if  train_params:
            var_mode = train_params['mode']
        else:
            unique_values, counts = np.unique(vec[~pd.isna(vec)], return_counts=True)
            var_mode = unique_values[np.argmax(counts)]
            self.params[name] = {'mode': var_mode}


        if impute:
            n_na = len(na_idx)
            if len(np.unique(var_mode)) == 1:
                vec[na_idx] = np.repeat(var_mode, n_na)
            else:
                vec[na_idx] = np.random.choice(var_mode, size=n_na, replace=True)

        return vec




def matchindex(d, t, k):
    """
    Find the indices of rows in a dataset (target cases) that have similar values to a set of donor cases.

    Parameters:
    - d: numpy array or list - The donor cases.
    - t: numpy array or list - The target cases.
    - k: int - Number of unique donors from which a random draw is made.

    Returns:
    - indices: numpy array - The indices of target cases that have similar values to the donor cases.
    """
    d = np.asarray(d)
    t = np.asarray(t)

    # Calculate the absolute difference between donor and target cases
    diff = np.abs(t.reshape(-1,1) - d)

    # Find the indices of the k nearest donors for each target case
    # argpartition is faster than sort as it is kind of a partial sort
    indices = np.argpartition(diff, k)[:,:k]

    selected_values = np.random.choice(indices.shape[1], size=indices.shape[0])

    # Get the selected values
    result = indices[np.arange(indices.shape[0]), selected_values]

    return result


def pmm_numeric_binary(yhatobs,yhatmis,yobs,k):
    # PMM for numeric or binary variable

    # Parameters:
    # yhatobs The predicted values of observed entries in a variable
    # yhatmis The predicted values of missing entries in a variable
    # yobs The actual observed values of observed entries in a variable
    # k The number of donors.

    #r = matchindex(d=yhatobs, t=yhatmis, k=k)

    yhatobs = np.asarray(yhatobs).reshape(-1,)
    yhatmis = np.asarray(yhatmis)

    # Calculate the absolute difference between donor and target cases

    diff = np.abs(yhatmis.reshape(-1, 1) - yhatobs)


    # Find the indices of the k nearest donors for each target case
    # argpartition is faster than sort as it is kind of a partial sort
    indices = np.argpartition(diff, k)[:, :k]

    selected_values = np.random.choice(indices.shape[1], size=indices.shape[0])

    # Get the selected values
    result = indices[np.arange(indices.shape[0]), selected_values]

    return yobs.iloc[result].to_numpy()


def pmm_multiclass(yhatobs, yhatmis, yobs, k):
    # Function for multiclass Predictive Mean Matching (PMM)
    #
    # Parameters:
    # - yhatobs: The predicted values of observed entries in a variable
    # - yhatmis: The predicted values of missing entries in a variable
    # - yobs: The actual observed values of observed entries in a variable
    # - k: The number of donors.
    #
    # Returns:
    # - match_class: The matched observed values of all missing entries



    # Perform matching (using k-nearest neighbors)
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(yhatobs)
    _, indices = neigh.kneighbors(yhatmis)

    selected_values = np.random.choice(indices.shape[1], size=indices.shape[0])

    # Get the selected values
    result = indices[np.arange(indices.shape[0]), selected_values]


    return yobs.iloc[result].to_numpy()





def createNA(data, var_names=None, p=0.3, random_seed=42):
    # this function creates missing values under MCAR(Missing Complete At Random) mechanism
    # data      - is a dataframe
    # var_names - is a list of columns where missing variable will be generated
    # p          - proportion of missing values in the dataframe. To control different missing proportion per column name
    #              pass a list
    # random_seed - for reproducability

    # make a copy of the dataframe
    data = data.copy()
    dtypes = data.dtypes
    n_row, n_col = data.shape
    if type(p) == list:
        n_p = len(p)
    else:
        n_p = 1

    np.random.seed(random_seed)

    if var_names is None:

        if n_p == 1:
            # Create a mask of random indices with True for elements to be set as missing
            mask = np.random.random((n_row, n_col)) < p
            # Set the elements in the data array corresponding to True in the mask as NaN
            data = data.where(~mask)
        elif n_p == n_col:
            if any(p > 1):
                raise ValueError("Value(s) greater than 1 passed for p")
            for i in range(n_col):
                mask = np.random.random((n_row, )) < p
                # Set the elements in the data array corresponding to True in the mask as NaN
                data.iloc[:, i] = data.iloc[:, i].where(~mask)
        else:
            raise ValueError(
                "When var_names is not specified, the length of p should be either one or data.shape[1] .")

    else:

        missing_vars = data.columns[data.isna().any()]
        k = len(var_names)

        if n_p == 1:
            p = {var: p for var in var_names}
        elif n_p != k:
            raise ValueError("The length of p should be either one or the same as the length of var_names.")
        else:
            p = {var: x for var, x in zip(var_names, p)}

        for var in var_names:
            if var in missing_vars:
                print(
                    f"Variable {var} has missing values in the original dataset. The proportion of missing values for this variable in the output data may be larger than the value specified in p.")

            mask = np.random.random((n_row, 1)) < p[var]
            # Set the elements in the data array corresponding to True in the mask as NaN
            data[var] = data[var].where(~mask.reshape(-1, ))

    # data types get changed when masking. pandas primarily uses NaN to represent missing data.
    # Because NaN is a float, this forces an array of integers with any missing values to become floating point.
    # convert to the original data type, or something equivalent. it is not possible
    # to have a column with dtype = np.int64. Change it to pandas type Int64
    for col in dtypes.index:
        str_d = str(dtypes[col])
        if 'int' == str_d.lower()[:3]:
            data[col] = data[col].astype("Int64")

    return data

