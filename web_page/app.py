from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
  
        ocean_proximity = data.get('ocean_proximity')
        median_income = float(data.get('median_income'))
        households = float(data.get('households'))
        population = float(data.get('population'))
        total_bedrooms = float(data.get('total_bedrooms'))
        total_rooms = float(data.get('total_rooms'))
        housing_median_age = float(data.get('housing_median_age'))
        latitude = float(data.get('latitude'))
        longitude = float(data.get('longitude'))

        
        model = joblib.load('model.pkl')  
        pipeline = joblib.load('pipeline.pkl')  
        features = [{"ocean_proximity ":str(ocean_proximity), "median_income": median_income, "households": households, "population": population, "total_bedrooms": total_bedrooms, "total_rooms": total_rooms, "housing_median_age": housing_median_age, "latitude": latitude, "longitude": longitude}]
        features_dataframe = pd.DataFrame(features)
        processed_features = pipeline.transform(features_dataframe)
        feature_names = pipeline.get_feature_names_out()
        prepared_df = pd.DataFrame(processed_features, columns=feature_names)

        prediction = model.predict(prepared_df)[0]
                
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
