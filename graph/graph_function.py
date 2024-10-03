from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery
from graph.consts import RETRIEVE, GENERATE, GRADE_DOCUMENTS, WEBSEARCH
from graph.state import GraphState


def decide_to_generate(state):
    """

    Determines whether to generate documents based on the current state.

    Parameters:
    state (dict): A dictionary containing the current state information.
                  It must include a key named "web_search".

    Returns:
    str: Returns 'WEBSEARCH' if the "web_search" key in the state is True.
         Otherwise, returns 'GENERATE'.
    """
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDED")
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """
    Evaluates the given textual generation from a graph state for grounding in documents
    and relevance to a given question. Returns a string indicating the usefulness of
    the generation based on the evaluation.

    Args:
        state (GraphState): A graph state containing 'question', 'documents', and 'generation'.

    Returns:
        str: An evaluation result which can be 'useful', 'not useful', or 'not supported'.
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION VS QUESTION")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def route_question(state: GraphState) -> str:
    """

        Routes a question to the appropriate data source based on its content.

        Args:
            state (GraphState): The current state containing the question to be routed.

        Returns:
            str: The destination to which the question has been routed. Either WEBSEARCH or RETRIEVE.
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE
