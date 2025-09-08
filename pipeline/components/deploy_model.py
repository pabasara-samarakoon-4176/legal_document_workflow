from kfp.dsl import component, Input, Model

@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform"]
)
def deploy_model(model_artifact: Input[Model],
                 project: str,
                 region: str,
                 endpoint_display_name: str = "llm-endpoint",
                 deploy: bool = True):
    from google.cloud import aiplatform as aip
    aip.init(project=project, location=region)

    model = aip.Model.upload(
        display_name="fine-tuned-llm",
        artifact_uri=model_artifact.uri,
        serving_container_image_uri=f"us-central1-docker.pkg.dev/{project}/llmops-repo/serve:latest",
        serving_container_ports=[8008],
        serving_container_predict_route="/predict",
        serving_container_health_route="/"
    )
    if not deploy:
        print("Model registered (not deployed).")
        return
    
    endpoint = aip.Endpoint.create(display_name=endpoint_display_name)
    model.deploy(endpoint=endpoint, machine_type="n1-standard-8", accelerator_type="NVIDIA_TESLA_T4", accelerator_count=1)
    print(f"Deployed to endpoint: {endpoint.resource_name}")