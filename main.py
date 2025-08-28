import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import requests
import json
import numpy as np


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


# Update: Send request to FastAPI proxy endpoint with API key
url = 'http://localhost:8000/predict'
headers = {
	"content-type": "application/json",
	"x-api-key": "my-secret-key"  # Must match the key in proxy_api.py
}

json_response = requests.post(url, data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']

predicted_classes = [1 if pred[0] > 0.5 else 0 for pred in predictions]
actual_classes = y_test.values.tolist()[0:5]
print("\nPredicted classes:")
print(predicted_classes)
print("Actual classes: "); 
print(actual_classes)