from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step, Context, InputRequiredEvent, HumanResponseEvent
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from llama_index.llms.openai import OpenAI
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.readers.file.markdown import MarkdownReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import json
import logging
import os
from dotenv import load_dotenv
import asyncio
import re
from rag_pipeline import search_clauses
from google.cloud import aiplatform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Please set the HF_TOKEN environment variable.")

PROJECT_ID = "legal-llmops-pipeline-1"
REGION = "us-central1"
ENDPOINT_DISPLAY_NAME = "llm-endpoint"

def get_llm_client(project: str, region: str, endpoint_display_name: str):
    """
    Initializes and returns a client for a custom LLM deployed on a Vertex AI endpoint.

    Args:
        project (str): The Google Cloud project ID.
        region (str): The region where the endpoint is located (e.g., "us-central1").
        endpoint_display_name (str): The display name of the deployed endpoint.

    Returns:
        A callable function that can be used to make predictions on the LLM.
    """
    aiplatform.init(project=project, location=region)
    
    # Find the endpoint by its display name.
    # The list() method returns a list of Endpoint objects that match the filter.
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_display_name}"')
    
    if not endpoints:
        raise ValueError(f"No endpoint found with display name: {endpoint_display_name}")
    
    endpoint = endpoints[0]

    def predict(prompt: str) -> str:
        """
        Sends a prediction request to the LLM endpoint.

        The structure of the `instances` payload depends on the custom
        serving container. This example assumes a simple JSON payload
        with a "prompt" key, which is a common pattern for LLMs.

        Args:
            prompt (str): The text prompt for the LLM.

        Returns:
            str: The generated text response from the model.
        """
        try:
            instances = [{"prompt": prompt}]

            response = endpoint.predict(instances=instances)
        
            prediction = response.predictions[0]

            return str(prediction)
            
        except Exception as e:
            return f"An error occurred during prediction: {e}"

    return predict

class LegalStartEvent(StartEvent):
    """
    Start event for initiating the legal document assistant workflow.
    """
    document_path: Path
    source: str = "unknown"

class DocumentIngestorEvent(Event):
    """
    Triggered after document ingestion and parsing.
    
    Attributes:
        document_path: Paths of ingested legal document
        document_id: Unique IDs assigned
        ingestion_timestamp: When ingestion occurred
        document_metadata: Title, length, parties, etc.
        source: Source of the upload
    """
    document_path: Path
    document_id: str
    ingestion_timestamp: datetime
    document_metadata: Dict[str, Any]
    source: str = "unknown"

class ClassifierEvent(Event):
    """
    Triggered after document classification.

    Attributes:
        document_id: ID of the classified document
        document_type: Detected legal type (NDA, contract, etc.)
        classification_confidence: Confidence in classification
        legal_categories: List of tags/ontologies
        timestamp: Time of classification
    """
    document_id: str
    document_type: str
    classification_confidence: float
    legal_categories: List[str]
    timestamp: datetime

class ClauseExtractorEvent(Event):
    """
    Triggered after clause extraction from document.

    Attributes:
        document_id: ID of the document
        clauses: Dict of clause_type -> text
        missing_clauses: List of expected but missing clauses
        extraction_timestamp: When extraction occurred
    """
    document_id: str
    clauses: Dict[str, str]
    missing_clauses: List[str]
    extraction_timestamp: datetime

class RAGEvent(Event):
    """
    Triggered after RAG-based search and augmentation.

    Attributes:
        document_id: ID of the document
        clause_recommendations: Suggested alternative/improved clauses
        source_documents: Supporting legal precedents
        similarity_scores: Similarity of clauses to known samples
        knowledge_sources: Datasets or embeddings used
    """
    document_id: str
    clause_recommendations: Dict[str, str]
    source_documents: List[Dict[str, Any]]
    similarity_scores: Dict[str, float]
    knowledge_sources: List[str]

class RiskAssessorEvent(Event):
    """
    Triggered after legal risk and compliance assessment.

    Attributes:
        document_id: ID of the document
        risk_scores: Clause-wise or overall scores
        flagged_issues: List of issues (missing, conflicting, outdated clauses)
        compliance_summary: Overview of legal or policy alignment
    """
    document_id: str
    risk_scores: Dict[str, float]
    flagged_issues: List[str]
    compliance_summary: str

class ValidatorEvent(Event):
    """
    Triggered before finalizing edits, allowing HITL intervention.

    Attributes:
        document_id: Document needing review
        flagged_clauses: Clause names needing approval
        reason: Why review is required
    """
    document_id: str
    flagged_clauses: List[str]
    reason: str

class ReportGeneratorEvent(Event):
    """
    Triggered when generating the final legal report.

    Attributes:
        document_id: ID of the document
        report_format: md, PDF, docx
        summary: Brief legal summary
        final_clauses: Modified clause content
        flagged_issues: Issues covered
        suggestions: Recommended changes
    """
    document_id: str
    report_path: Path
    report_format: str
    summary: str
    final_clauses: Dict[str, str]
    flagged_issues: List[str]
    suggestions: List[str]
    timestamp: datetime

class AuditLoggerEvent(Event):
    """
    Logs all decisions and revisions for audit trail.

    Attributes:
        document_id: ID of the document
        actions: List of agent/HITL actions with timestamps
        review_logs: Notes or flags during validation
    """
    document_id: str
    actions: List[Dict[str, Any]]
    review_logs: Optional[List[str]]

class FeedbackEvent(Event):
    """
    Captures post-analysis feedback from a user or reviewer.
    """
    document_id: str
    feedback: str

class ProgressEvent(Event):
    """Event for tracking workflow progress."""
    stage: str
    message: str
    progress_percent: float
    document_id: Optional[str] = None

class LegalDocumentWorkflow(Workflow):
    """
    Legal Document Assistant Workflow

    This workflow automates the processing, analysis, and reporting of legal documents
    (e.g., contracts, NDAs, agreements) using a modular agentic architecture augmented
    with human-in-the-loop (HITL) capabilities.

    Stages:
        1. Parse (Document Ingestor): Ingests uploaded markdown-based legal documents
           and extracts basic metadata such as title, size, and content length.
        
        2. Classify (Classifier): Classifies the type of legal document (e.g., NDA, 
           employment contract) and identifies legal domains or ontologies.
        
        3. Extract (Clause Extractor): Parses and extracts individual clauses or sections
           from the document, tagging missing or ambiguous ones.
        
        4. Retrieve (RAG): Performs Retrieval-Augmented Generation to compare and enhance
           clauses using external knowledge (e.g., legal precedent, clause banks).
        
        5. Assess Risk (Risk Assessor): Scores the document and its clauses for compliance,
           risk exposure, missing terms, or misalignments with best practices.
        
        6. Validate (Validator): Flags uncertain sections or decisions for human validation
           and integrates feedback using InputRequiredEvent and HumanResponseEvent.
        
        7. Report (Report Generator): Generates a structured legal analysis report in
           markdown or PDF format with summaries, recommendations, and scores.
        
        8. Audit (Audit Logger): Logs all critical actions and human responses for 
           regulatory and transparency purposes.
        
        9. Stop: Final step to return the full results, outputs, and quality metadata.

    Human-in-the-Loop:
        - Enabled through specialized events like InputRequiredEvent and HumanResponseEvent
          to support dynamic review, validation, and revision of AI-generated content.

    Use Cases:
        - Contract compliance checking
        - Legal clause standardization
        - Intelligent clause suggestions and negotiation prep
        - Internal document audits or external legal readiness checks

    Technologies:
        - Agentic RAG (with LLMs + Vector Search)
        - Document parsing and classification
        - Async event-based execution
        - OpenAI embeddings, ChromaDB for retrieval
        - Markdown-based reporting and clause manipulation

    """

    llm = OpenAI(model="gpt-4o-mini-2024-07-18", api_key=api_key)
    # llm = HuggingFaceInferenceAPI(
    #     model_name="deepseek-ai/DeepSeek-V3-0324",
    #     token=hf_token,
    #     provider="auto",
    # )
    # llm = get_llm_client(project=PROJECT_ID, region=REGION, endpoint_display_name=ENDPOINT_DISPLAY_NAME)

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        super().__init__(**kwargs)

    @step
    async def process_uploaded_documents(
        self,
        ctx: Context,
        ev: LegalStartEvent
    ) -> DocumentIngestorEvent:
        """Ingests uploaded documents and extracts basic metadata."""
        print("====================================================================================")
        ctx.write_event_to_stream(ProgressEvent(
            stage="ingestion",
            message="Ingesting legal documents...",
            progress_percent=5.0
        ))
        doc_path = ev.document_path
        if not doc_path or not doc_path.exists():
            raise ValueError("Provided document path does not exist.")

        if not doc_path.suffix.lower().endswith(".md"):
            raise ValueError(f"Unsupported file type: {doc_path.name}. Only .md files are supported.")

        try:
            reader = MarkdownReader()
            docs = reader.load_data(str(doc_path))
            if not docs:
                raise ValueError("No content loaded from the document.")

            doc = docs[0]
            doc_id = f"legal_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            metadata = {
                "title": doc.metadata.get("title", doc_path.stem),
                "file_path": str(doc_path),
                "file_size": doc_path.stat().st_size,
                "text_length": len(doc.text),
                "doc_id": doc_id
            }

            await ctx.store.set(f"document_{doc_id}", doc)

            return DocumentIngestorEvent(
                document_path=doc_path,
                document_id=doc_id,
                ingestion_timestamp=datetime.now(),
                document_metadata=metadata,
                source=getattr(ev, "source", "file_upload")
            )

        except Exception as e:
            self.logger.error(f"Failed to process document {doc_path}: {e}")
            raise RuntimeError(f"Failed to ingest document: {e}")
        
    @step
    async def classify_documents(
        self,
        ctx: Context,
        ev: DocumentIngestorEvent
    ) -> ClassifierEvent:
        """Classifies document type and legal categories."""
        print("====================================================================================")
        ctx.write_event_to_stream(ProgressEvent(
            stage="classification",
            message="Classification uploaded legal documents...",
            progress_percent=10.0,
            document_id=ev.document_id
        ))   

        doc_id = ev.document_id
        doc = await ctx.store.get(f"document_{doc_id}")
        doc_text = doc.text if hasattr(doc, "text") else str(doc)

        prompt = f"""
        You are a legal document classifier. Read the following document content and classify it:

        Document Text:
        \"\"\"
        `{doc_text[:2000]}  # truncate for token safety
        \"\"\"`

        Respond in JSON with the following fields:
        - document_type: The type of legal document (e.g., NDA, Employment Agreement)
        - legal_categories: List of legal domains or tags (e.g., Confidentiality, IP, Termination)
        - classification_confidence: Float between 0 and 1
        """

        try:
            response = await self.llm.acomplete(prompt)
            response_text = response.text
            cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", response_text).strip()
            print(cleaned)
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse LLM response, using fallback values.")
            parsed = {
                "document_type": "Unknown",
                "legal_categories": [],
                "classification_confidence": 0.0
            }
        return ClassifierEvent(
            document_id=doc_id,
            document_type=parsed["document_type"], 
            classification_confidence=parsed["classification_confidence"],
            legal_categories=parsed["legal_categories"],
            timestamp=datetime.now(),
        )
    
    @step
    async def extract_clauses(
        self, 
        ctx: Context,
        ev: ClassifierEvent
    ) -> ClauseExtractorEvent:
        """Extracts clauses and identifies missing sections."""
        print("====================================================================================")
        ctx.write_event_to_stream(ProgressEvent(
            stage="extraction",
            message="Extracting clauses using LLM...",
            progress_percent=20.0,
            document_id=ev.document_id
        ))

        doc_key = f"document_{ev.document_id}"
        document = await ctx.store.get(doc_key)
        if document is None:
            raise ValueError(f"No document found in context with key: {doc_key}")
        
        doc_text = document.text

        prompt = f"""
        You are a legal document assistant. Given the following legal document, extract key clauses
        and identify any missing important clauses such as termination, indemnity, or dispute resolution.

        Return the response as a JSON with:
        - "clauses": A dictionary mapping clause names to text.
        - "missing_clauses": A list of clause types that are not found in the document.

        Document:
        \"\"\"
        {doc_text}
        \"\"\"

        Respond in JSON format only.
        """

        try:
            response = await self.llm.acomplete(prompt)
            response_text = response.text
            cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", response_text).strip()
            print(cleaned)
            parsed = json.loads(cleaned)

            clauses = parsed.get("clauses", {})
            missing_clauses = parsed.get("missing_clauses", [])

            return ClauseExtractorEvent(
                document_id=ev.document_id,
                clauses=clauses,
                missing_clauses=missing_clauses,
                extraction_timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Failed to extract clauses from document {ev.document_id}: {e}")
            raise RuntimeError(f"Clause extraction failed: {e}")
    
    @step
    async def retreive_context(
        self,
        ctx: Context,
        ev: ClauseExtractorEvent
    ) -> RAGEvent:
        """Retrieves external context and recommends clause improvements."""
        print("====================================================================================")
        ctx.write_event_to_stream(ProgressEvent(
            stage="retrieval",
            message="Retrieving similar clauses from external knowledge sources...",
            progress_percent=35.0,
            document_id=ev.document_id
        ))
        
        # clause_recommendations = {}
        # similarity_scores = {}
        # source_documents = []
        # knowledge_sources = ["Mock legal precident database"]

        # for clause_name, clause_text in ev.clauses.items():
        #     retrieved_example = f"This is a stronger version of the {clause_name.replace('_', ' ')} based on best practices."
        #     clause_recommendations[clause_name] = retrieved_example
        #     similarity_scores[clause_name] = 0.82
        #     source_documents.append({
        #         "title": "Legal Precedent",
        #         "content": f"Example content for {clause_name}."
        #     })

        clause_recommendations = {}
        similarity_scores = {}
        source_documents = []
        knowledge_sources = ["Legal Precedent Vector DB"]

        for clause_name, clause_text in ev.clauses.items():
            retrieved_items = search_clauses(clause_text)
            if retrieved_items:
                top_item = retrieved_items[0]
                clause_recommendations[clause_name] = top_item.get("text", "")
                similarity_scores[clause_name] = top_item.get("similarity", 0.0)

                for item in retrieved_items:
                    source_documents.append({
                        "title": item["title"],
                        "section": item.get("section", ""),
                        "content": item["text"],
                        "source": item.get("source", "")
                    })
            else:
                clause_recommendations[clause_name] = clause_text
                similarity_scores[clause_name] = 0.0

        await ctx.store.set(f"updated_clauses_{ev.document_id}", clause_recommendations)

        return RAGEvent(
            document_id=ev.document_id,
            clause_recommendations=clause_recommendations,
            source_documents=source_documents,
            similarity_scores=similarity_scores,
            knowledge_sources=knowledge_sources
        )
    
    @step
    async def assess_risk(
        self,
        ctx: Context,
        ev: RAGEvent | FeedbackEvent
    ) -> RiskAssessorEvent:
        """
        Assesses legal risk and compliance of document clauses using clause recommendations
        or human feedback. Uses an LLM to simulate scoring and flagging risks.

        Returns per-clause risk scores, flagged issues, and a high-level compliance summary.
        """
        print("====================================================================================")
        ctx.write_event_to_stream(ProgressEvent(
            stage="assessment",
            message="Assessing legal risk and compliance...",
            progress_percent=50.0,
            document_id=ev.document_id
        ))

        document_id = getattr(ev, "document_id", "unknown_doc")

        if isinstance(ev, RAGEvent):
            clause_data = ev.clause_recommendations
            context_data = ev.source_documents
        elif isinstance(ev, FeedbackEvent):
            clause_data = await ctx.store.get(f"updated_clauses_{ev.document_id}") or {}
            context_data = []
        else:
            raise ValueError("Unsupported event type passed to assess_risk")

        formatted_clauses = "\n".join([f"Clause: {k}\nContent: {v}" for k, v in clause_data.items()])
        context_blurb = "\n".join([f"{doc['title']}: {doc['content']}" for doc in context_data])

        prompt = f"""
        You are a legal compliance assistant.

        Below are clauses extracted from a legal document along with recommended improvements.
        Also included are reference clauses from a legal precedent knowledge base.

        --- Clauses ---
        {formatted_clauses}

        --- Context ---
        {context_blurb}

        Please:
        1. Identify any clauses that may pose legal risk or need revision.
        2. Assign a risk score between 0.0 (no risk) and 1.0 (high risk) for each clause.
        3. Flag missing or ambiguous sections.
        4. Provide a 1-sentence compliance summary.
        
        Respond the following in JSON format:
        - risk_scores: {{clause_name: float, ...}}
        - flagged_issues: [ ... ]
        - compliance_summary: "..."
        """

        response = await self.llm.acomplete(prompt)
        response_text = response.text
        cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", response_text).strip()
        print(cleaned)
        output_text = json.loads(cleaned)

        try:
            risk_scores = output_text.get("risk_scores", {})
            flagged_issues = output_text.get("flagged_issues", [])
            compliance_summary = output_text.get("compliance_summary", "No summary provided.")

        except Exception as e:
            self.logger.warning(f"Failed to parse LLM risk assessment output. Defaulting to fallback. Error: {e}")
            risk_scores = {k: 0.5 for k in clause_data.keys()}
            flagged_issues = ["Unable to extract structured flags."]
            compliance_summary = "Partial compliance - unable to verify all clauses."

        return RiskAssessorEvent(
            document_id=document_id,
            risk_scores=risk_scores,
            flagged_issues=flagged_issues,
            compliance_summary=compliance_summary
        )
    
    @step
    async def get_feedback(
        self,
        ctx: Context,
        ev: RiskAssessorEvent
    ) -> InputRequiredEvent:
        """
        Requests human feedback on risk assessment results by summarizing key risks, 
        flagged clauses, and asking for clarification or validation.
        """
        print("====================================================================================")
        ctx.write_event_to_stream(ProgressEvent(
            stage="validaition",
            message="Requesting human feedback on risk assessment validation...",
            progress_percent=65.0,
            document_id=ev.document_id
        ))

        await ctx.store.set(f"risk_assessment_{ev.document_id}", {
            "risk_scores": ev.risk_scores,
            "flagged_issues": ev.flagged_issues,
            "compliance_summary": ev.compliance_summary
        })

        return InputRequiredEvent(
            document_id=ev.document_id,
            prefix="Please review the risk assessment results and provide feedback.\n"
        )
    
    @step
    async def recv_feedback(
        self,
        ctx: Context,
        ev: HumanResponseEvent
    ) -> FeedbackEvent | ReportGeneratorEvent:
        """
        Processes human feedback or proceeds directly to report generation.
        Feedback may lead to a re-assessment or report finalization.
        """
        print("====================================================================================")
        ctx.write_event_to_stream(ProgressEvent(
            stage="feedback",
            message="Capturing feedback from user...",
            progress_percent=75.0,
            document_id=ev.document_id
        ))

        assessment = await ctx.store.get(f"risk_assessment_{ev.document_id}")
        if assessment is None:
            self.logger.warning(f"No previous risk assessment found for document: {ev.document_id}")
            assessment = {}

        system_prompt = f"""
        You are an expert legal workflow assistant.
        Based on the following human feedback, decide whether more revision is needed
        (i.e., return 'revise') or the feedback approves the analysis (i.e., return 'proceed').

        Feedback:
        \"\"\"{ev.feedback.strip()}\"\"\"

        Output one word only: either 'revise' or 'proceed'.
        """

        response = (await self.llm.acomplete(system_prompt))
        decision_text = getattr(response, "text", str(response)).strip().lower()
        print("Decision output:", decision_text)
        if "revise" in decision_text:
            return FeedbackEvent(
                document_id=ev.document_id,
                feedback=ev.feedback
            )
        else:
            return ReportGeneratorEvent(
                document_id=ev.document_id,
                report_path=Path(f"report_{ev.document_id}.md"),
                report_format="markdown",
                summary="Legal analysis completed with human review.",
                final_clauses={"confidentiality_clause": "Finalized clause after review."},
                flagged_issues=assessment.get("flagged_issues", []),
                suggestions=["Ensure completeness of termination clause"],
                timestamp=datetime.now()
            )
    
    @step
    async def generate_report(
        self,
        ctx: Context,
        ev: ReportGeneratorEvent
    ) -> AuditLoggerEvent:
        """Generates the final legal analysis report."""
        print("====================================================================================")
        ctx.write_event_to_stream(ProgressEvent(
            stage="generation",
            message="Generating final legal report...",
            progress_percent=90.0,
            document_id=ev.document_id
        ))

        # Build the Markdown content
        report_md = f"# Legal Analysis Report for Document: {ev.document_id}\n\n"

        report_md += f"**Generated On:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        report_md += "## 📝 Executive Summary\n"
        report_md += f"{ev.summary}\n\n"

        report_md += "## 📄 Finalized Clauses\n"
        for clause_name, text in ev.final_clauses.items():
            report_md += f"### {clause_name.replace('_', ' ').title()}\n{text}\n\n"

        report_md += "## ⚠️ Flagged Issues\n"
        if ev.flagged_issues:
            for i, issue in enumerate(ev.flagged_issues, 1):
                report_md += f"{i}. {issue}\n"
        else:
            report_md += "No issues flagged.\n"
        report_md += "\n"

        report_md += "## ✅ Recommendations\n"
        for i, suggestion in enumerate(ev.suggestions, 1):
            report_md += f"{i}. {suggestion}\n"
        report_md += "\n"

        # Save to file
        output_path = ev.report_path or Path(f"report_{ev.document_id}.md")
        with open(output_path, "w", encoding="utf-8") as f:
            print(f"Writing report to {output_path}")
            f.write(report_md)

        await ctx.store.set(f"report_path_{ev.document_id}", str(output_path))

        return AuditLoggerEvent(
            document_id=ev.document_id,
            actions=[{"action": "Report generated", "timestamp": datetime.now().isoformat()}],
            review_logs=["Reviewed confidentiality clause", "Generated markdown report"]
        )
    
    @step
    async def log_audit(
        self,
        ctx: Context,
        ev: AuditLoggerEvent
    ) -> StopEvent:
        """Logs audit trail and completes the workflow."""
        print("====================================================================================")
        ctx.write_event_to_stream(ProgressEvent(
            stage="logging",
            message="Logging audit trail...",
            progress_percent=100.0,
            document_id=ev.document_id
        ))

        audit_entry = {
            "document_id": ev.document_id,
            "timestamp": datetime.now().isoformat(),
            "actions": ev.actions,
            "review_logs": ev.review_logs
        }
        audit_log_path = Path(f"audit_logs/audit_{ev.document_id}.jsonl")
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry) + "\n")
        await ctx.store.set(f"audit_log_{ev.document_id}", audit_entry)

        return StopEvent(
            outputs={
                "document_id": ev.document_id,
                "status": "Workflow completed successfully.",
                "audit_log_path": str(audit_log_path)
            }
        )
    
async def test_llm_output():
    llm = OpenAI(model="gpt-4o-mini-2024-07-18", api_key=api_key)
    prompt = "Say hello and return a JSON object: {\"greeting\": \"Hello, world!\"}"
    response = await llm.acomplete(prompt)
    parsed = json.loads(response.strip())
    print(parsed)
    
async def main():
    document_path = "input_documents/nda_sample.md"
    start_event = LegalStartEvent(document_path=document_path)
    workflow = LegalDocumentWorkflow(timeout=60, verbose=True)
    draw_all_possible_flows(workflow, filename="legal_workflow_flow.html")
    handler = workflow.run(start_event=start_event)
    async for event in handler.stream_events():
        print(event)
        if isinstance(event, InputRequiredEvent):
            response = input(event.prefix)
            handler.ctx.send_event(
                HumanResponseEvent(
                    document_id=event.document_id,
                    feedback=response
                )
            )
    result = await handler
    # print("Workflow result: ", result)

asyncio.run(main())