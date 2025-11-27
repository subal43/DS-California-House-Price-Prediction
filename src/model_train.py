import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import json
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor 

MODEL_FILE = "model.pkl"
PIPELINE = "pipeline.pkl" 

def build_classifier_from_params(params):
    classifier_name = params["classifier"]
    hp = {k: v for k, v in params.items() if k != "classifier"}

    if classifier_name == "LightGBM":
        model = LGBMRegressor(**hp)
    elif classifier_name == "RandomForest":
        model = RandomForestRegressor(**hp)
    elif classifier_name == "LinearRegression":
        model = LinearRegression(**hp)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    return model






def build_pipeline(num_attribs , cat_attribs):
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
        ('num', numerical_pipeline, num_attribs),
        ('cat', categorical_pipeline, cat_attribs),
    ])
    return full_pipeline

if not (os.path.exists(MODEL_FILE ) and os.path.exists(PIPELINE)):
    print("Training Model and Preparing Pipeline...")
    data = pd.read_csv('./data/housing.csv')
    print("Data Loaded")
    # Creating income category attribute for stratified sampling
    data['income_cat'] = pd.cut(data['median_income'],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
    # Stratified sampling based on income category
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data['income_cat']):
        strat_train_set = data.loc[train_index].drop('income_cat', axis=1)
    

    # Separating features and labels
    housing = strat_train_set.drop('median_house_value', axis=1)
    housing_labels = strat_train_set['median_house_value'].copy()

    
    #list of numerical and categorical attributes
    numerical_attrs = housing.select_dtypes(include=[np.number]).columns.tolist()
    categorical_attrs = housing.select_dtypes(include=[object]).columns.tolist()

    # Preparing the pipeline
    full_pipeline = build_pipeline(numerical_attrs , categorical_attrs)
    # Preparing the data
    housing_prepared = full_pipeline.fit_transform(housing)
    
    # Training the model
    # model = RandomForestRegressor(random_state=42)
    # model.fit(housing_prepared, housing_labels)

    with open('./optuna_result/best_hyperparameters.json', 'r') as f:
        best_params = json.load(f)

    


    
# ====== Example Usage ======
 

    model = build_classifier_from_params(best_params["best_params"])
    model.fit(housing_prepared, housing_labels)
  


    # lgb_reg = lgb.LGBMRegressor(random_state=42)
    # lgb_reg.fit(housing_prepared, housing_labels)
    
    # Saving the model and pipeline
    joblib.dump(  model , MODEL_FILE)  
    joblib.dump(full_pipeline, PIPELINE)

    print("Model and Pipeline saved.")

else : 
    print("Model and Pipeline already exist.")
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE)

    input_data = pd.read_csv('input.csv')
    feature_names = pipeline.get_feature_names_out()
    prepared_data = pipeline.transform(input_data)
    prepared_df = pd.DataFrame(prepared_data, columns=feature_names)

    predictions = model.predict(prepared_df)
    input_data['Predicted_House_Value'] = predictions
    input_data.to_csv('predictions_lgb_reg_2.csv', index=False)
    print("Predictions saved to Predicted_House_Value.csv")


