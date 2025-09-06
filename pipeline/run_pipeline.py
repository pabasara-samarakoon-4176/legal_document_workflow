# run_pipeline.py
from google.cloud import aiplatform

PROJECT = "legal-llmops-pipeline"
REGION  = "us-central1"
PIPELINE_ROOT = "gs://legal-pipeline/pipeline_root"
JSONL_PATH    = "gs://legal-pipeline/dataset_v1.jsonl"
SERVICE_ACCOUNT = "llmops-pipeline@{}.iam.gserviceaccount.com".format(PROJECT)

aiplatform.init(project=PROJECT, location=REGION, staging_bucket=PIPELINE_ROOT)

job = aiplatform.PipelineJob(
    display_name="llmops-finetune-jsonl",
    template_path="llmops_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        "project": PROJECT,
        "region": REGION,
        "pipeline_root": PIPELINE_ROOT,
        "jsonl_path": JSONL_PATH,
        "deploy_threshold": 0.25
    }
)

# Important: run with the pipeline runner SA
job.run(service_account=SERVICE_ACCOUNT, sync=True)
