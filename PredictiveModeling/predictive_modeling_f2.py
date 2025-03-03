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

