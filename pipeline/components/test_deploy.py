from google.cloud import aiplatform as aip

project = "legal-llmops-pipeline-1"
region = "us-central1"
bucket_model_uri = "gs://legal-llmops-pipeline-1/gpt2-model/"

aip.init(project=project, location=region)

# Upload model
model = aip.Model.upload(
    display_name="gpt2",
    artifact_uri=bucket_model_uri,
    serving_container_image_uri="us-central1-docker.pkg.dev/legal-llmops-pipeline-1/llmops-repo/gpt2-serve:latest",
    serving_container_ports=[8080],
    serving_container_predict_route="/predict",
    serving_container_health_route="/"
)

print("Model uploaded:", model.resource_name)

endpoint = model.deploy(
    deployed_model_display_name="gpt2-deployed",
    machine_type="n1-standard-4",
    traffic_split={"0": 100},
)
print("Endpoint deployed:", endpoint.resource_name)