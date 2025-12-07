import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
import base64
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

model = joblib.load('model.pkl')  
pipeline = joblib.load('pipeline.pkl')  

@app.route('/')
def index():
    return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if data is None:
            return jsonify({'error': 'No input data provided'}), 400
  
        ocean_proximity = str(data.get('ocean_proximity'))
        median_income = float(data.get('median_income'))
        households = float(data.get('households'))
        population = float(data.get('population'))
        total_bedrooms = float(data.get('total_bedrooms'))
        total_rooms = float(data.get('total_rooms'))
        housing_median_age = float(data.get('housing_median_age'))
        latitude = float(data.get('latitude'))
        longitude = float(data.get('longitude'))

        

        features = [{"ocean_proximity":ocean_proximity, "median_income": median_income, "households": households, "population": population, "total_bedrooms": total_bedrooms, "total_rooms": total_rooms, "housing_median_age": housing_median_age, "latitude": latitude, "longitude": longitude}]
        features_dataframe = pd.DataFrame(features)
        processed_features = pipeline.transform(features_dataframe)
        feature_names = pipeline.get_feature_names_out()
        prepared_df = pd.DataFrame(processed_features, columns=feature_names)

        prediction = model.predict(prepared_df)[0]
        try:
            plot_url = None
        
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(prepared_df)


            if isinstance(shap_values, list):
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values[0] if shap_values.ndim == 2 else shap_values

            
            fig , ax = plt.subplots(figsize=(10,6))

            feature_names_list = list (prepared_df.columns)
            feature_values = shap_vals

            indices = sorted(range(len(feature_values)), key=lambda i: abs(feature_values[i]), reverse=True)[:10]
            sorted_feature_names = [feature_names_list[i] for i in indices]
            sorted_feature_values = [feature_values[i] for i in indices]

            colors = ['#ef4444' if val < 0 else '#22c55e' for val in sorted_feature_values]
            y_pos = range(len(sorted_feature_names))

            ax.barh(y_pos, sorted_feature_values, color=colors , alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_feature_names)
            ax.set_xlabel('SHAP Values (Impact on Prediction)',fontsize=12, fontweight='bold')
            ax.set_title('Feature Importance for This Prediction', fontsize=14, fontweight='bold')
            plt.tight_layout()
            ax.axvline(0, color='black', linewidth=0.8)
            ax.grid(axis='x',alpha=0.3)

            for i, v in enumerate(sorted_feature_values):
                ax.text(v + (0.02 if v > 0 else -0.02), i, f"{v:.3f}", color='black', va='center', fontweight='bold', fontsize=9)
            
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi =100 , bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
            plt.close(fig)
            plot_url = f'data:image/png;base64,{plot_url}'
        
        except Exception as e:
            print(f"SHAP plot generation failed: {e}")
            pass
        
        response = {'prediction': float(prediction)}
        if plot_url:
            response['plot_url'] = plot_url
        
        return jsonify(response)    

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
