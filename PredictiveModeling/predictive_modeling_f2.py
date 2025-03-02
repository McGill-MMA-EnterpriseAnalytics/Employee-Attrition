#!/usr/bin/env python
# coding: utf-8

# In[107]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import fbeta_score, make_scorer, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin, clone
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import os
from IPython.display import display, Markdown


# In[108]:


df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.drop(columns=['EmployeeCount', 'StandardHours', 'EmployeeNumber'], inplace=True)
df.drop_duplicates(inplace=True)

# Convert Attrition to binary: 'Yes' → 1, 'No' → 0.
df["Attrition"] = df["Attrition"].map({'Yes': 1, 'No': 0})

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# Reset indices to ensure alignment
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)


# In[109]:


display(X_train)


# In[110]:


# Identify numeric and categorical columns
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Columns to keep as ordinal (already integers)
ordinal_cols = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement',
                'JobLevel', 'JobSatisfaction', 'PerformanceRating',
                'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']

# For numeric columns, we want to apply log transform only on those that are not ordinal.
numeric_to_transform = [col for col in numeric_cols if col not in ordinal_cols]

# For categorical columns, separate BusinessTravel so we can experiment with its encoding.
business_travel = ['BusinessTravel']
other_cat_cols = [col for col in categorical_cols if col not in business_travel]


# In[111]:


class AddNewFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Adds three new features:
      - AgeAtJoining = Age - YearsAtCompany
      - TenureRatio = YearsAtCompany / TotalWorkingYears
      - IncomePerYearExp = MonthlyIncome / TotalWorkingYears
    """
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_ = X.copy()
        # Make sure these columns exist
        if 'Age' in X_.columns and 'YearsAtCompany' in X_.columns:
            X_['AgeAtJoining'] = X_['Age'] - X_['YearsAtCompany']
        if 'YearsAtCompany' in X_.columns and 'TotalWorkingYears' in X_.columns:
            # Avoid division by zero
            ratio = X_['YearsAtCompany'] / X_['TotalWorkingYears'].replace(0, np.nan)
            X_['TenureRatio'] = ratio.fillna(0)
        if 'MonthlyIncome' in X_.columns and 'TotalWorkingYears' in X_.columns:
            ratio2 = X_['MonthlyIncome'] / X_['TotalWorkingYears'].replace(0, np.nan)
            X_['IncomePerYearExp'] = ratio2.fillna(0)
        return X_


# In[ ]:


import numpy as np
import pandas as pd

class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mapping=None):
        # Default mapping if none is provided.
        if mapping is None:
            mapping = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
        self.mapping = mapping
        
    def fit(self, X, y=None):
        # Optionally, you could store the unique categories here.
        # For now, simply mark the encoder as fitted.
        self.fitted_ = True
        return self
    
    def transform(self, X):
        # Ensure the transformer has been fitted.
        if not hasattr(self, 'fitted_'):
            raise ValueError("This CustomOrdinalEncoder instance is not fitted yet. Call 'fit' before using this method.")
        
        # If X is a numpy array and has shape (n_samples, 1), flatten it.
        if isinstance(X, np.ndarray):
            if X.ndim == 2 and X.shape[1] == 1:
                s = pd.Series(X.ravel())
                mapped = s.map(self.mapping)
                if mapped.isnull().any():
                    raise ValueError("Some values were not found in the mapping")
                return mapped.values.reshape(-1, 1)
            else:
                raise ValueError("Expected a 2D array with a single column")
        else:
            # If X is a pandas DataFrame or Series.
            if isinstance(X, pd.DataFrame):
                if X.shape[1] == 1:
                    s = X.iloc[:, 0]
                    mapped = s.map(self.mapping)
                    if mapped.isnull().any():
                        raise ValueError("Some values were not found in the mapping")
                    return mapped.values.reshape(-1, 1)
                else:
                    # If there are multiple columns, apply mapping on each.
                    return X.apply(lambda col: col.map(self.mapping)).values
            elif isinstance(X, pd.Series):
                mapped = X.map(self.mapping)
                if mapped.isnull().any():
                    raise ValueError("Some values were not found in the mapping")
                return mapped.values.reshape(-1, 1)
            else:
                raise ValueError("Input type not recognized. Expected numpy array or pandas DataFrame/Series.")


# In[113]:


from imblearn.base import BaseSampler
from sklearn.ensemble import IsolationForest

class IsolationForestRemover(BaseSampler):
    """
    An outlier removal step using IsolationForest, implemented as an imblearn 'sampler'.
    This approach aligns with how ImbPipeline expects sampling methods to behave.
    """
    _parameter_constraints = {}
    _sampling_type = 'clean-sampling'

    def __init__(self, contamination=0.05, random_state=42):
        super().__init__()
        self.contamination = contamination
        self.random_state = random_state

    def _fit_resample(self, X, y):
        """
        The core method required by 'BaseSampler' subclasses.
        It fits the IsolationForest on X, then filters out rows (and corresponding y)
        flagged as outliers.
        """
        iso = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )
        iso.fit(X)
        preds = iso.predict(X)
        mask = (preds == 1)  # Keep only inliers
        return X[mask], y[mask]


# In[114]:


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from scipy.stats import boxcox

class BoxCoxSkewedTransformer(BaseEstimator, TransformerMixin):
    """
    Applies Box-Cox transformation only to the columns specified in skewed_cols.
    Automatically shifts data if min value <= 0.
    """
    def __init__(self, skewed_cols=None):
        if skewed_cols is None:
            skewed_cols = []
        self.skewed_cols = skewed_cols
        self.col_names_ = None

    def fit(self, X, y=None):
        # Capture column names if X is a DataFrame; otherwise, use generic names.
        if isinstance(X, pd.DataFrame):
            self.col_names_ = X.columns.tolist()
        else:
            self.col_names_ = [f"col_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        # Convert to DataFrame using stored column names
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.col_names_)
        else:
            X_df = X.copy()

        # Apply Box-Cox only to self.skewed_cols (if they exist in X_df)
        for col in self.skewed_cols:
            if col in X_df.columns:
                col_min = X_df[col].min()
                shift = 0
                if col_min <= 0:
                    shift = abs(col_min) + 1
                # boxcox returns a tuple (transformed, lambda), so we take index [0]
                X_df[col] = boxcox(X_df[col] + shift)[0]

        # Return values as a NumPy array to remain consistent with scikit-learn
        return X_df.values

    def get_feature_names_out(self, input_features=None):
        """
        If input_features is provided, return it (typical in ColumnTransformer),
        otherwise return the stored col_names_.
        """
        if input_features is not None:
            return np.array(input_features)
        return np.array(self.col_names_ if self.col_names_ is not None else [])


# In[115]:


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class LogTransformSkewed(BaseEstimator, TransformerMixin):
    def __init__(self, skewed_cols=None):
        if skewed_cols is None:
            skewed_cols = []
        self.skewed_cols = skewed_cols
        self.col_names_ = None

    def fit(self, X, y=None):
        # If X is a DataFrame, store its columns. Otherwise, just store generic names.
        if isinstance(X, pd.DataFrame):
            self.col_names_ = list(X.columns)
        else:
            self.col_names_ = [f"col_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        # Convert NumPy array to DataFrame using stored col names (if needed)
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.col_names_)
        else:
            X_df = X.copy()  # Already a DataFrame

        # Apply log transform to only the specified skewed columns
        for col in self.skewed_cols:
            if col in X_df.columns:
                X_df[col] = np.log1p(X_df[col])

        return X_df.values  # Return as array for downstream steps

    def get_feature_names_out(self, input_features=None):
        # So the pipeline can retrieve feature names if needed
        if input_features is not None:
            return np.array(input_features)
        return np.array(self.col_names_ if self.col_names_ is not None else [])


# In[116]:


def monkey_patch_get_signature_names_out():
    from inspect import Signature, signature, Parameter
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    import pandas as pd

    default_get_feature_names_out = StandardScaler.get_feature_names_out

    if not hasattr(SimpleImputer, "get_feature_names_out"):
        SimpleImputer.get_feature_names_out = default_get_feature_names_out

    if not hasattr(FunctionTransformer, "get_feature_names_out"):
        orig_init = FunctionTransformer.__init__
        orig_sig = signature(orig_init)

        def __init__(*args, feature_names_out=None, **kwargs):
            orig_sig.bind(*args, **kwargs)
            orig_init(*args, **kwargs)
            args[0].feature_names_out = feature_names_out

        __init__.__signature__ = Signature(
            list(signature(orig_init).parameters.values()) + [
                Parameter("feature_names_out", Parameter.KEYWORD_ONLY)]
        )

        def get_feature_names_out(self, names=None):
            if callable(self.feature_names_out):
                return self.feature_names_out(self, names)
            assert self.feature_names_out == "one-to-one"
            return default_get_feature_names_out(self, names)

        FunctionTransformer.__init__ = __init__
        FunctionTransformer.get_feature_names_out = get_feature_names_out

    if not hasattr(CustomOrdinalEncoder, "get_feature_names_out"):
        def custom_ordinal_get_feature_names_out(self, input_features=None):
            if input_features is not None and len(input_features) > 0:
                return np.array(input_features)
            else:
                return np.array(["custom_ordinal_encoded_feature"])
        CustomOrdinalEncoder.get_feature_names_out = custom_ordinal_get_feature_names_out

    if not hasattr(AddNewFeaturesTransformer, "get_feature_names_out"):
        def add_new_feats_get_feature_names_out(self, input_features=None):
            """
            After transformation, new columns are added. This is a simplistic approach 
            that just appends the new column names.
            """
            if input_features is None:
                return np.array(["Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
                                 "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
                                 "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
                                 "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
                                 "Over18", "OverTime", "PercentSalaryHike", "PerformanceRating",
                                 "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
                                 "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
                                 "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
                                 # The new columns:
                                 "AgeAtJoining", "TenureRatio", "IncomePerYearExp"])
            else:
                # If we started with known features, append the new feature names:
                return np.concatenate([input_features, 
                                       np.array(["AgeAtJoining", "TenureRatio", "IncomePerYearExp"])])
        AddNewFeaturesTransformer.get_feature_names_out = add_new_feats_get_feature_names_out

    if not hasattr(IsolationForestRemover, "get_feature_names_out"):
        def iso_remover_get_feature_names_out(self, input_features=None):
            if input_features is not None:
                return np.array(input_features)
            else:
                return np.array(["filtered_features"])
        IsolationForestRemover.get_feature_names_out = iso_remover_get_feature_names_out

monkey_patch_get_signature_names_out()


# In[117]:


# =========================
# 3. Define Helper Functions
# =========================


skewed_cols = [
    'IncomePerYearExp', 'YearsSinceLastPromotion', 'YearsAtCompany', 
    'MonthlyIncome', 'TotalWorkingYears', 'NumCompaniesWorked', 'DistanceFromHome', 
    'YearsInCurrentRole', 'PercentSalaryHike', 'YearsWithCurrManager'
]



log_transformer = LogTransformSkewed(skewed_cols=skewed_cols)
boxcox_transformer = BoxCoxSkewedTransformer(skewed_cols=skewed_cols)
no_transformer = 'passthrough'


# In[118]:


# Create a pipeline step to optionally remove outliers (IsolationForest) or do nothing.
outlier_step = [
    ('iso_remover', IsolationForestRemover()),  # or 'passthrough'
]

# Add new features
add_features_step = [
    ('add_new_feats', AddNewFeaturesTransformer())
]

# Numeric pipeline: median imputation -> log/boxcox/no transform -> scale (Standard or MinMax)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('transformer', log_transformer),  # Will param-grid over log_transformer, boxcox_transformer, 'passthrough'
    ('scaler', StandardScaler())       # Will param-grid over StandardScaler or MinMaxScaler
])

# Ordinal columns: pass them as-is
ordinal_pipeline = 'passthrough'

# BusinessTravel: OneHot or CustomOrdinal
bus_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first'))  # We'll param-grid over OneHot or CustomOrdinal
])

# 4f. Other categorical columns: OneHot
other_cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first'))
])

# 4g. Build the overall preprocessor
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_to_transform),
    ('ord', ordinal_pipeline, ordinal_cols),
    ('bus', bus_pipeline, business_travel),
    ('other_cat', other_cat_pipeline, other_cat_cols)
])





# In[119]:


feat_sel_rfe = RFE(estimator=LogisticRegression(max_iter=500, random_state=42), n_features_to_select=10)
feat_sel_lasso = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=42))
feat_sel_tree = SelectFromModel(RandomForestClassifier(random_state=42), threshold='median')


# In[ ]:


# Create an F2 scorer: beta=2 gives more weight to recall.
f2_scorer = make_scorer(fbeta_score, beta=2)

# Define the base models with fixed (default) hyperparameters.
base_models = {
    'LogisticRegression': LogisticRegression(max_iter=500, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'MLP': MLPClassifier(max_iter=500, random_state=42)
}

# Our full pipeline: optional outlier removal -> add new features -> preprocessor -> SMOTE -> feature selection -> classifier
full_pipeline = ImbPipeline([
    ('add_feats', AddNewFeaturesTransformer()),
    ('preprocessor', clone(preprocessor)),
    ('outlier', 'passthrough'),        # param-grid over IsolationForestRemover or pass
    ('smote', SMOTE(random_state=42)),
    ('feat_sel', 'passthrough'),
    ('classifier', LogisticRegression())  # placeholder, replaced per model
])

# Param grid for pipeline steps (not the classifier's internal hyperparams yet)
pipeline_param_grid = {
    'outlier': [IsolationForestRemover(contamination=0.05, random_state=42), 'passthrough'],
    'preprocessor__num__transformer': [log_transformer, boxcox_transformer, no_transformer],
    'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler()],
    'preprocessor__bus__encoder': [OneHotEncoder(drop='first'), CustomOrdinalEncoder()],
    'feat_sel': ['passthrough', feat_sel_rfe, feat_sel_lasso, feat_sel_tree]
}

# Dictionary to store the best pipeline for each base model (without classifier hyperparameter tuning)
best_pipeline_nonhyper = {}

best_pipeline_nonhyper = {}
for model_name, model in base_models.items():
    print(f"\nTuning pipeline steps for {model_name}...")
    
    # Clone the full pipeline and replace the classifier
    pipeline_model = clone(full_pipeline)
    pipeline_model.set_params(classifier=model)
    
    grid_search_model = GridSearchCV(
        pipeline_model,
        param_grid=pipeline_param_grid,
        cv=5,
        scoring=f2_scorer,
        n_jobs=-1,
        verbose=2
    )
    
    grid_search_model.fit(X_train, y_train)
    
    best_pipeline = grid_search_model.best_estimator_
    best_pipeline_nonhyper[model_name] = best_pipeline
    print(f"Best pipeline for {model_name}: {grid_search_model.best_params_}")


# In[121]:


def display_cv_f2_recall_precision_roc(pipeline_dict, X, y, cv=5):
    scoring = {
        'f2': f2_scorer,
        'recall': 'recall',
        'precision': 'precision',
        'roc_auc': 'roc_auc',
        'accuracy': 'accuracy' 
    }
    
    for model_name, pipeline in pipeline_dict.items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        mean_f2 = scores['test_f2'].mean()
        std_f2 = scores['test_f2'].std()
        mean_recall = scores['test_recall'].mean()
        std_recall = scores['test_recall'].std()
        mean_precision = scores['test_precision'].mean()
        std_precision = scores['test_precision'].std()
        mean_roc_auc = scores['test_roc_auc'].mean()
        std_roc_auc = scores['test_roc_auc'].std()
        mean_accuracy = scores['test_accuracy'].mean()  # Access accuracy results
        std_accuracy = scores['test_accuracy'].std()
        
        print(f"{model_name}:")
        print(f"  Mean CV F2 Score     = {mean_f2:.4f} (Std: {std_f2:.4f})")
        print(f"  Mean CV Recall       = {mean_recall:.4f} (Std: {std_recall:.4f})")
        print(f"  Mean CV Precision    = {mean_precision:.4f} (Std: {std_precision:.4f})")
        print(f"  Mean CV ROC-AUC      = {mean_roc_auc:.4f} (Std: {std_roc_auc:.4f})")
        print(f"  Mean CV Accuracy     = {mean_accuracy:.4f} (Std: {std_accuracy:.4f})")
        print("-" * 40)
display_cv_f2_recall_precision_roc(best_pipeline_nonhyper, X_train, y_train)


# In[ ]:


# Define grids for each classifier
classifier_grids = {
    'LogisticRegression': {
         'classifier__C': [0.01, 0.1, 1, 10, 100],
         'classifier__penalty': [None, 'l1', 'l2'],
         'classifier__solver': ['liblinear', 'saga']
    },
    'DecisionTree': {
         'classifier__max_depth': [None, 5, 10, 20],
         'classifier__min_samples_split': [2, 5, 10],
         'classifier__min_samples_leaf': [1, 2, 4],
         'classifier__criterion': ['gini', 'entropy']
    },
    'RandomForest': {
         'classifier__n_estimators': [100, 200, 300],
         'classifier__max_depth': [None, 5, 10, 20],
         'classifier__min_samples_split': [2, 5, 10],
         'classifier__min_samples_leaf': [1, 2, 4],
         'classifier__max_features': ['sqrt', 'log2', None]
    },
    'GradientBoosting': {
         'classifier__n_estimators': [100, 200, 300],
         'classifier__learning_rate': [0.01, 0.1, 1],
         'classifier__max_depth': [3, 5, 7, 10],
         'classifier__subsample': [0.8, 1.0],
         'classifier__max_features': ['sqrt', 'log2', None]
    },
    'MLP': {
         'classifier__hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
         'classifier__alpha': [0.0001, 0.001, 0.01],
         'classifier__learning_rate_init': [0.001, 0.01],
         'classifier__activation': ['relu', 'tanh']
    }
}

best_classifier_pipelines_random = {}
n_iter = 50  # adjust as needed

for clf_name, clf_grid in classifier_grids.items():
    print(f"\nTuning classifier hyperparameters with random search: {clf_name}")
    
    # Create new pipeline using the best pipeline steps found for this classifier
    best_pipe_steps = best_pipeline_nonhyper[clf_name]
    
    # Build a new pipeline using those steps, but we replace the classifier hyperparams
    tuned_pipeline = ImbPipeline([
        ('add_feats', best_pipe_steps.named_steps['add_feats']),
        ('preprocessor', best_pipe_steps.named_steps['preprocessor']),
        ('outlier', best_pipe_steps.named_steps['outlier']),
        ('smote', SMOTE(random_state=42)),
        ('feat_sel', best_pipe_steps.named_steps['feat_sel']),
        ('classifier', clone(base_models[clf_name]))
    ])
    

    random_search = RandomizedSearchCV(
         tuned_pipeline,
         param_distributions=clf_grid,
         n_iter=n_iter,
         cv=5,
         scoring=f2_scorer,
         n_jobs=-1,
         verbose=2,
         random_state=42
    )
    random_search.fit(X_train, y_train)
    
    best_classifier_pipelines_random[clf_name] = random_search.best_estimator_
    print(f"Best parameters for {clf_name}:")
    print(random_search.best_params_)
    print(f"Best CV F2 Score for {clf_name}: {random_search.best_score_:.4f}")


# In[123]:


def display_best_pipeline_details(pipeline_dict, X_train, y_train, X_test, y_test, cv=5):
    scoring = {
        'f2': f2_scorer,
        'precision': 'precision',
        'recall': 'recall',
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc'
    }
    print("\n===== Best Pipeline Details for Each Model =====\n")
    for model_name, pipeline in pipeline_dict.items():
        params = pipeline.get_params()
        print(f"Model: {model_name}")
        print("Pipeline parameters:")
        print(f"  outlier: {params.get('outlier')}")
        print(f"  preprocessor__num__transformer: {params.get('preprocessor__num__transformer')}")
        print(f"  preprocessor__num__scaler: {params.get('preprocessor__num__scaler')}")
        print(f"  preprocessor__bus__encoder: {params.get('preprocessor__bus__encoder')}")
        print(f"  feat_sel: {params.get('feat_sel')}")
        print(f"  classifier: {params.get('classifier')}")

        # Print classifier hyperparams
        clf_params = {k: v for k, v in params.items() if k.startswith('classifier__')}
        if clf_params:
            print("  Classifier Hyperparameters:")
            for k, v in clf_params.items():
                print(f"    {k}: {v}")

        cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        mean_f2 = cv_results['test_f2'].mean()
        std_f2 = cv_results['test_f2'].std()
        mean_precision = cv_results['test_precision'].mean()
        std_precision = cv_results['test_precision'].std()
        mean_recall = cv_results['test_recall'].mean()
        std_recall = cv_results['test_recall'].std()
        mean_accuracy = cv_results['test_accuracy'].mean()
        std_accuracy = cv_results['test_accuracy'].std()
        mean_roc_auc = cv_results['test_roc_auc'].mean()
        std_roc_auc = cv_results['test_roc_auc'].std()

        test_score = pipeline.score(X_test, y_test)
        print("Performance Metrics (CV):")
        print(f"  Mean F2 Score  = {mean_f2:.4f} (Std: {std_f2:.4f})")
        print(f"  Mean Precision = {mean_precision:.4f} (Std: {std_precision:.4f})")
        print(f"  Mean Recall    = {mean_recall:.4f} (Std: {std_recall:.4f})")
        print(f"  Mean Accuracy  = {mean_accuracy:.4f} (Std: {std_accuracy:.4f})")
        print(f"  Mean ROC AUC   = {mean_roc_auc:.4f} (Std: {std_roc_auc:.4f})")
        print(f"Test set score (default metric): {test_score:.4f}")
        print("="*50 + "\n")

display_best_pipeline_details(best_classifier_pipelines_random, X_train, y_train, X_test, y_test)


# In[124]:


estimators = [(name, pipe) for name, pipe in best_classifier_pipelines_random.items()]
meta_model = LogisticRegression(max_iter=500, random_state=42)

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=5,
    passthrough=False
)

stacking_param_grid = {
    'final_estimator__C': [0.01, 0.1, 1, 10, 100],
    'final_estimator__penalty': ['l2'],
    'final_estimator__solver': ['liblinear', 'saga'],
    'passthrough': [False]
}

scoring_dict = {'f2': f2_scorer, 'roc_auc': 'roc_auc'}
n_iter = 20

stacking_search_lr = RandomizedSearchCV(
    stacking_clf,
    param_distributions=stacking_param_grid,
    n_iter=n_iter,
    cv=5,
    scoring=scoring_dict,
    refit='f2',
    n_jobs=-1,
    verbose=2,
    random_state=42
)
stacking_search_lr.fit(X_train, y_train)

print("Best stacking parameters (Logistic Meta-Model):")
print(stacking_search_lr.best_params_)
print("Best stacking CV F2-score:", stacking_search_lr.best_score_)

best_index = stacking_search_lr.best_index_
mean_cv_roc_auc = stacking_search_lr.cv_results_['mean_test_roc_auc'][best_index]
print("Best stacking CV ROC-AUC:", mean_cv_roc_auc)

y_proba = stacking_search_lr.best_estimator_.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_proba)
print("Test ROC-AUC for best stacking ensemble:", test_roc_auc)
stacking_test_accuracy = stacking_search_lr.best_estimator_.score(X_test, y_test)
print("Test set accuracy for best stacking ensemble:", stacking_test_accuracy)


# In[125]:


os.makedirs("models", exist_ok=True)

# Save each tuned base classifier
for model_name, model in best_classifier_pipelines_random.items():
    filename = os.path.join("models", f"{model_name}_tuned_model.joblib")
    joblib.dump(model, filename)
    print(f"Saved {model_name} tuned model to {filename}")

# Save stacking (logistic meta-model)
stacking_lr_filename = os.path.join("models", "stacking_logistic_regression.joblib")
joblib.dump(stacking_search_lr.best_estimator_, stacking_lr_filename)
print(f"Saved stacking logistic regression ensemble to {stacking_lr_filename}")

