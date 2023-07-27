import pandas as pd
import joblib
from flask import Flask, request, jsonify

data_file = "plantDiseases.csv"
df = pd.read_csv(data_file)


model = joblib.load("plantModel.joblib")

label_encoder = joblib.load("label_encoder.joblib")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_disease():
    data = request.json
    name = data.get("name")
    ageCat = data.get("ageCat")

    name_encoded = label_encoder.transform([name])[0]

    if name_encoded in df['name'].values:

        user_input = pd.DataFrame({
            'name': [name_encoded],
            'ageCat': [ageCat]
        })

      
        predicted_disease = model.predict(user_input)[0]


        disease_data = df[df['disease'] == predicted_disease].iloc[0].to_dict()

 
        disease_data['name'] = label_encoder.inverse_transform([disease_data['name']])[0]


        response = {
            "predictions": [
                {
                    "predDisease": disease_data['disease'],
                    "otherInfo": {
                        "Timeline": disease_data['Timeline'],
                        "Causes": disease_data['Causes'],
                        "Symptoms": disease_data['Symptoms'],
                        "Treatment&remedies": disease_data['Treatment&remedies'],
                        "Prevention": disease_data['Prevention']
                    }
                }
            ]
        }
        return jsonify(response)

    else:
        return jsonify({"error": "Invalid plant name"}), 400

if __name__ == '__main__':
    app.run(debug=True)
