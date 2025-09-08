# run_pipeline.py
from google.cloud import aiplatform

PROJECT = "legal-llmops-pipeline-1"
REGION  = "us-central1"
BUCKET_NAME = "legal-llmops-pipeline-1"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"
JSONL_PATH    = f"gs://{BUCKET_NAME}/datasets/dataset_v1.jsonl"
TEST_PATH    = f"gs://{BUCKET_NAME}/datasets/test_v1.jsonl"

aiplatform.init(
    project=PROJECT, 
    location=REGION, 
    staging_bucket=PIPELINE_ROOT
)

job = aiplatform.PipelineJob(
    display_name="llmops-finetune-jsonl",
    template_path="llmops_pipeline.json",
    pipeline_root=f"gs://{BUCKET_NAME}/pipeline_root",
    parameter_values={
        "project": PROJECT,
        "region": REGION,
        "jsonl_path": JSONL_PATH,
        "deploy_threshold": 0.25
    },
    # enable_caching=False
)

job.run(sync=True)
