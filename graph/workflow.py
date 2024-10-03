from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from graph.consts import RETRIEVE, GENERATE, GRADE_DOCUMENTS, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from graph.graph_function import decide_to_generate

load_dotenv()


def build_workflow(add_grade_generation=False, conditional_entry_point=False):
    """
    This function constructs a workflow for processing based on given options.

    Parameters:
    - add_grade_generation (bool): If True, additional conditional edges for grade generation will be added.
    - conditional_entry_point (bool): If True, the entry point will be conditionally set based on the routing of the question.

    Returns:
    - StateGraph: A configured workflow graph.

    The workflow comprises common nodes: RETRIEVE, GRADE_DOCUMENTS, GENERATE, and WEBSEARCH.

    - Entry Point:
      - If conditional_entry_point is True, a conditional entry point is set based on the routing question.
      - Otherwise, the entry point is set to RETRIEVE.

    - Common Edges:
      - An edge from RETRIEVE to GRADE_DOCUMENTS.
      - Conditional edges from GRADE_DOCUMENTS to either WEBSEARCH or GENERATE, based on the decision function.
      - An edge from WEBSEARCH to GENERATE, and an edge from GENERATE to END.

    - Optional Additional Conditional Edges:
      - If add_grade_generation is True, additional conditional edges for grade generation will be added from GENERATE, based on the grade generation function.
    """
    from graph.graph_function import (
        grade_generation_grounded_in_documents_and_question,
        route_question,
    )

    workflow = StateGraph(GraphState)

    # Add common nodes
    workflow.add_node(RETRIEVE, retrieve)
    workflow.add_node(GRADE_DOCUMENTS, grade_documents)
    workflow.add_node(GENERATE, generate)
    workflow.add_node(WEBSEARCH, web_search)

    # Set entry point
    if conditional_entry_point:
        workflow.set_conditional_entry_point(
            route_question, {WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE}
        )
    else:
        workflow.set_entry_point(RETRIEVE)

    # Add common edges
    workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
    workflow.add_conditional_edges(
        GRADE_DOCUMENTS,
        decide_to_generate,
        {
            WEBSEARCH: WEBSEARCH,
            GENERATE: GENERATE,
        },
    )
    workflow.add_edge(WEBSEARCH, GENERATE)
    workflow.add_edge(GENERATE, END)

    # Optionally add additional conditional edges
    if add_grade_generation:
        workflow.add_conditional_edges(
            GENERATE,
            grade_generation_grounded_in_documents_and_question,
            {
                "not supported": GENERATE,
                "useful": END,
                "not useful": WEBSEARCH,
            },
        )

    return workflow
