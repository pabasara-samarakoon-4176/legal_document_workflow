from kfp import dsl
from kfp.compiler import Compiler
from components.preprocess_jsonl import preprocess_jsonl
from components.fine_tune_llm import fine_tune_llm
from components.evaluate_llm import evaluate_llm
from components.deploy_model import deploy_model

@dsl.pipeline(
    name="llmops-finetune-jsonl",
    description="Preprocess JSONL, fine-tune LLM (LoRA), evaluate, and (optionally) deploy."
)
def llmops_pipeline(project: str,
                    region: str,
                    pipeline_root: str,
                    jsonl_path: str,
                    deploy_threshold: float = 0.25):
    pre = preprocess_jsonl(jsonl_path=jsonl_path)
    trn = fine_tune_llm(preprocessed_data=pre.outputs["preprocessed_output"])
    score = evaluate_llm(preprocessed_data=pre.outputs["preprocessed_output"],
                         model_dir=trn.outputs["model_output"])

    with dsl.If(score.output >= deploy_threshold):
        deploy_model(model_artifact=trn.outputs["model_output"],
                     project=project, region=region, deploy=True)

Compiler().compile(pipeline_func=llmops_pipeline, package_path="llmops_pipeline.json")