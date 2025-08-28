import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import requests
import json
import argparse

# Canary test script: compares predictions between two model versions via FastAPI proxy.
# This script does NOT promote or demote models; it only reports comparison results.

print("Preparing test data for canary test...")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
test_samples = X_test_scaled[0:10]

parser = argparse.ArgumentParser()
parser.add_argument('--version_a', type=int, required=True, help='Old model version')
parser.add_argument('--version_b', type=int, required=True, help='New model version')
args = parser.parse_args()

url = f'http://localhost:8000/canary?version_a={args.version_a}&version_b={args.version_b}'
headers = {
    "content-type": "application/json",
    "x-api-key": "my-secret-key"
}
data_payload = json.dumps({"instances": test_samples.tolist()})

try:
    response = requests.post(url, data=data_payload, headers=headers, timeout=10)
    response.raise_for_status()
    result = response.json()
    preds_a = result.get('version_a', [])
    preds_b = result.get('version_b', [])
    print(f"Predictions from version {args.version_a}: {preds_a}")
    print(f"Predictions from version {args.version_b}: {preds_b}")
    matches = sum([int(round(a[0]) == round(b[0])) for a, b in zip(preds_a, preds_b)])
    print(f"Number of matching predictions: {matches} out of {len(preds_a)}")
    # Add more sophisticated checks here if needed
except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON response: {e}")
    print(f"Response content: {getattr(response, 'text', None)}")
except Exception as e:
    print(f"Unexpected error: {e}")
