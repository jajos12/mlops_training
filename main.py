import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import requests
import json
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("API_KEYS", "my-secret-key").split(",")[0].strip()

print("loading dataset .....")
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

test_samples = X_test_scaled[0:5]

data = json.dumps({"instances": test_samples.tolist()})


url = 'http://localhost:8000/predict'
headers = {
    "content-type": "application/json",
    "x-api-key": API_KEY
}

try:
    response = requests.post(url, data=data, headers=headers, timeout=10)
    response.raise_for_status()
    result = response.json()
    predictions = result.get('predictions')
    if predictions is None:
        print("Error: 'predictions' key not found in response.")
        print("Response content:", result)
    else:
        predicted_classes = [1 if pred[0] > 0.5 else 0 for pred in predictions]
        actual_classes = y_test.values.tolist()[0:5]
        print("\nPredicted classes:")
        print(predicted_classes)
        print("Actual classes: ")
        print(actual_classes)
except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON response: {e}")
    print(f"Response content: {getattr(response, 'text', None)}")