## 1. Endpoint and API Documentation

### **URL Structure**
The endpoint for making predictions on your deployed model is:
`http://localhost:8501/v1/models/my_model:predict`

-   `http://localhost:8501`: This is the host and port of your running TensorFlow Serving instance.
-   `/v1/models`: This is the standard path for the TensorFlow Serving REST API.
-   `/my_model`: This is the **model name** (`MODEL_NAME`) you set when running the Docker container.
-   `:predict`: This is the **verb** used to request an inference from the model.

---

## 2. Sample Request Code

The following Python script demonstrates how to prepare data and send a POST request to your model endpoint. It also includes error handling by converting the NumPy array to a JSON-serializable list.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import requests
import json
import numpy as np

# Load and prepare data (same as your script)
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

# Prepare the JSON payload by converting the NumPy array to a list
data_payload = json.dumps({"instances": test_samples.tolist()})

# Define the endpoint and headers
url = 'http://localhost:8501/v1/models/my_model:predict'
headers = {"content-type": "application/json"}

# Send the POST request
try:
    json_response = requests.post(url, data=data_payload, headers=headers)
    json_response.raise_for_status() # Raise an exception for bad status codes
    predictions = json.loads(json_response.text)['predictions']
    predicted_classes = [pred[0] for pred in predictions]
    actual_classes = y_test.values.tolist()[0:5]

    print("\nPredicted classes:")
    print(predicted_classes)
    print("Actual classes: ")
    print(actual_classes)

except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON response: {e}")
    print(f"Response content: {json_response.text}")
except KeyError:
    print("Response does not contain 'predictions' key.")
```

---

## 3. Promotion and Rollback Logic

- The training script (`train.py`) checks if the new model's accuracy is at least 97%.
- If the new model passes, it is saved to a new versioned folder (e.g., `exports/2/`). TensorFlow Serving will serve the highest version.
- If the new model fails, the previous version remains active. No rollback is needed, as the old version is never deleted unless explicitly done.

---

## 4. Monitoring with Prometheus

TensorFlow Serving exposes Prometheus metrics at `/monitoring/prometheus` (default port 8501).

- To enable monitoring, run TensorFlow Serving with the default settings.
- Example Prometheus scrape config:
  ```yaml
  scrape_configs:
    - job_name: 'tensorflow_serving'
      static_configs:
        - targets: ['localhost:8501']
  ```
- Access metrics at: `http://localhost:8501/monitoring/prometheus`

---

## 5. Canary Testing

To perform a canary test:
- Send test requests to the new model version endpoint (e.g., `/v1/models/my_model/versions/2:predict`).
- Compare predictions with the previous version to ensure consistency and quality.
- Example Python script is provided below.

---

## 6. Automation with GitHub Actions

You can automate retraining and deployment using GitHub Actions. Example workflow:

```yaml
name: Retrain and Promote Model
on:
  schedule:
    - cron: '0 0 * * 0'  # Runs weekly on Sunday at midnight
  workflow_dispatch:

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run training script
        run: |
          python train.py
      - name: Upload exported model
        uses: actions/upload-artifact@v3
        with:
          name: exported-model
          path: exports/
```

---

## 7. Canary Test Script Example

```python
import requests
import json
import numpy as np

# Prepare test data (replace with your own test samples)
test_samples = np.random.rand(5, 30)  # Example shape for breast cancer dataset

# Predict with previous version
url_prev = 'http://localhost:8501/v1/models/my_model/versions/1:predict'
headers = {"content-type": "application/json"}
data = json.dumps({"instances": test_samples.tolist()})
response_prev = requests.post(url_prev, data=data, headers=headers)
preds_prev = json.loads(response_prev.text)['predictions']

# Predict with new version
url_new = 'http://localhost:8501/v1/models/my_model/versions/2:predict'
response_new = requests.post(url_new, data=data, headers=headers)
preds_new = json.loads(response_new.text)['predictions']

print('Previous version predictions:', preds_prev)
print('New version predictions:', preds_new)
```

---
