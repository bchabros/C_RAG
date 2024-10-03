from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from graph.consts import RETRIEVE, GENERATE, GRADE_DOCUMENTS, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from graph.graph_function import decide_to_generate

load_dotenv()


def build_workflow(add_grade_generation=False, conditional_entry_point=False):
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
