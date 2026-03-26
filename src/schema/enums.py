from enum import Enum


class QueryType(str, Enum):
    DEFINITIONAL = "definitional"      # "What is X?"
    PROCEDURAL = "procedural"          # "What steps to do X?"
    STRUCTURAL = "structural"          # "How do A and B relate?"
    COMPLIANCE = "compliance"          # "Am I violating X?"
    TEMPORAL = "temporal"              # "When does X take effect?"


class RetrievalStrategy(str, Enum):
    VECTOR_SEARCH = "vector_search"
    GRAPH_QUERY = "graph_query"
    CROSS_REFERENCE = "cross_reference"
    SUB_QUESTION = "sub_question"
    HIERARCHICAL = "hierarchical"
    PROPOSITIONAL = "propositional"
    HYBRID = "hybrid"


class Confidence(str, Enum):
    HIGH = "high"        # multiple corroborating sources
    MEDIUM = "medium"    # single authoritative source
    LOW = "low"          # ambiguous or insufficient data


class FactType(str, Enum):
    RULE = "rule"
    EXCEPTION = "exception"
    PENALTY = "penalty"
    CONDITION = "condition"
    DEFINITION = "definition"
