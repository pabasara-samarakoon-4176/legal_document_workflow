from google.cloud import aiplatform

project = "390498257074"
region = "us-central1"
endpoint_id = "8445677566924161024"

endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{project}/locations/{region}/endpoints/{endpoint_id}")

response = endpoint.predict([{"inputs": "Hello, world!"}])
print("Prediction response:", response)