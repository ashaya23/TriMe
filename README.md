# TriMe

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