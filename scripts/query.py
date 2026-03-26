"""CLI script to query the Agentic RAG system."""

import argparse
import sys

from src.graph.builder import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question to the Agentic RAG system."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="The legal question to answer.",
    )
    args = parser.parse_args()

    print(f"Question: {args.question}\n")

    try:
        graph = build_graph()
        result = graph.invoke(
            {"original_query": args.question, "messages": []},
        )
    except Exception as exc:
        print(f"Query failed: {exc}", file=sys.stderr)
        sys.exit(1)

    answer = result.get("synthesis", "")
    confidence = result.get("confidence")
    citations = result.get("citations", [])

    print("Answer:")
    print(answer)

    if confidence:
        conf_val = confidence.value if hasattr(confidence, "value") else confidence
        print(f"\nConfidence: {conf_val}")

    if citations:
        print(f"\nCitations ({len(citations)}):")
        for i, cite in enumerate(citations, 1):
            ref = cite.section_ref if hasattr(cite, "section_ref") else cite.get("section_ref", "")
            print(f"  [{i}] {ref}")


if __name__ == "__main__":
    main()
