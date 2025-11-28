import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import optuna
import json 
import lightgbm as lgb

#Loading the dataset
data1 = pd.read_csv('./data/housing.csv')
data = data1.copy()

# Creating income category attribute for stratified sampling
data['income_cat'] = pd.cut(data['median_income'],
                            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                            labels=[1, 2, 3, 4, 5])
# Stratified sampling based on income category
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['income_cat']):
    strat_train_set = data.loc[train_index].drop('income_cat', axis=1)
    strat_test_set = data.loc[test_index].drop('income_cat', axis=1)

# Separating features and labels
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

#list of numerical and categorical attributes
numerical_attrs = housing.select_dtypes(include=[np.number]).columns.tolist()
categorical_attrs = housing.select_dtypes(include=[object]).columns.tolist()

# Numerical pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ('encoding',OneHotEncoder(handle_unknown='ignore'))
])

# Full pipeline
full_pipeline = ColumnTransformer([
    ('num', numerical_pipeline, numerical_attrs),
    ('cat', categorical_pipeline, categorical_attrs),
])

# Preparing the data
housing_prepared = full_pipeline.fit_transform(housing)


def objective(trial) : 
    classfier_name = trial.suggest_categorical("classifier" , ['RandomForest' , 'LightGBM', 'DecisionTree' , 'LinearRegression'])
    if classfier_name == 'RandomForest' :
        n_estimators = trial.suggest_int("n_estimators" , 50 , 300)
        max_depth = trial.suggest_int("max_depth" , 5 , 30)
        min_samples_split = trial.suggest_int("min_samples_split" , 2 , 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf" , 1 , 4)
        bootstrap = trial.suggest_categorical("bootstrap" , [True , False])
        model = RandomForestRegressor(n_estimators=n_estimators , max_depth=max_depth ,
                                      min_samples_split=min_samples_split , min_samples_leaf=min_samples_leaf ,
                                      bootstrap=bootstrap , random_state=42)
       

    elif classfier_name == 'LightGBM' :
        num_leaves = trial.suggest_int("num_leaves" , 20 , 150)
        learning_rate = trial.suggest_float("learning_rate" , 0.01 , 0.3)
        model = lgb.LGBMRegressor(num_leaves=num_leaves , learning_rate=learning_rate , random_state=42)

    elif classfier_name == 'DecisionTree' :
        max_depth = trial.suggest_int("max_depth" , 5 , 30)
        model = DecisionTreeRegressor( max_depth=max_depth , random_state=42)

    elif classfier_name == 'LinearRegression' :
        model = LinearRegression() 

   
    feature_names = full_pipeline.get_feature_names_out()
    input_data = pd.DataFrame(housing_prepared, columns=feature_names)
    scores = -cross_val_score(model, input_data, housing_labels,
                                        scoring='neg_mean_squared_error', cv=5).mean()
    return scores


study = optuna.create_study(direction='minimize')
study.optimize(objective , n_trials=100)

best_trial = study.best_trial
best_results = {
    "best_params": best_trial.params,
    "best_value": best_trial.value
}

with open("./optuna_result/best_hyperparameters_1.json" , "w") as f :
    json.dump(best_results , f , indent=4)
study.trials_dataframe().to_csv("./optuna_result/optuna_study_results_1.csv", index=False)
