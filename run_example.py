"""
ScholarSync — Example Run Script

Demonstrates how to use the pipeline programmatically (without the API).
"""

import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from scholarsync.config.settings import get_settings
from scholarsync.ingestion.pdf_loader import load_pdf
from scholarsync.ingestion.chunker import chunk_document
from scholarsync.rag.vector_store import add_chunks
from scholarsync.workflow.langgraph_pipeline import run_pipeline
from scholarsync.utils.schemas import PaperMetadata


def main():
    settings = get_settings()

    # ── Check configuration ─────────────────────────────────────
    if not settings.groq_api_key:
        print("❌ ERROR: GROQ_API_KEY not set.")
        print("   Copy .env.example to .env and add your Groq API key.")
        sys.exit(1)

    print("=" * 60)
    print("  ScholarSync — Agentic AI Literature Review")
    print("=" * 60)

    # ── Step 1: Load and process PDFs ───────────────────────────
    pdf_dir = Path(settings.upload_dir)
    if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
        print(f"\n📁 No PDFs found in {pdf_dir}")
        print("   Place your research papers (PDFs) in the data/uploads/ directory.")
        print("   Then re-run this script.")
        sys.exit(0)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"\n📄 Found {len(pdf_files)} PDF(s):")

    paper_metadata_list: list[PaperMetadata] = []

    for pdf_file in pdf_files[:settings.max_papers]:
        print(f"   Loading: {pdf_file.name}")
        paper_meta, pages = load_pdf(pdf_file)
        chunks = chunk_document(paper_meta, pages)
        paper_meta.total_chunks = len(chunks)
        add_chunks(chunks)
        paper_metadata_list.append(paper_meta)
        print(f"   → {len(pages)} pages, {len(chunks)} chunks indexed")

    # ── Step 2: Run the pipeline ────────────────────────────────
    query = input("\n🔍 Enter your research query: ").strip()
    if not query:
        query = "Provide a comprehensive literature review comparing methodologies, key findings, and research gaps across all papers."

    print(f"\n🚀 Starting pipeline for: \"{query}\"")
    print("-" * 60)

    session_id = f"example_{os.urandom(4).hex()}"
    final_state = run_pipeline(session_id, query, paper_metadata_list)

    # ── Step 3: Display results ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    # Progress
    for msg in final_state.get("progress_messages", []):
        print(f"  {msg}")

    # Errors
    if final_state.get("errors"):
        print(f"\n⚠️ Errors:")
        for err in final_state["errors"]:
            print(f"   - {err}")

    # Report
    report_md = final_state.get("report_markdown", "")
    if report_md:
        print(f"\n{'═' * 60}")
        print("  LITERATURE REVIEW REPORT")
        print(f"{'═' * 60}\n")
        print(report_md)

        # Save to file
        report_path = Path(settings.reports_dir) / f"{session_id}_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_md, encoding="utf-8")
        print(f"\n💾 Report saved to: {report_path}")
    else:
        print("\n❌ No report generated.")


if __name__ == "__main__":
    main()
