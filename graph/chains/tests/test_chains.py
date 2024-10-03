from dotenv import load_dotenv
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_grader, GraderHallucinations
from graph.chains.router import question_router, RouteQuery
from ingestion import retriever

load_dotenv()


def test_retrival_grader_answer_yes() -> None:
    """
    Test function to verify if the retrieval grader correctly grades the given document as relevant ('yes') for the specified question.

    test_retrival_grader_answer_yes() -> None

    question: Question string for agent memory.
    docs: List of documents retrieved based on the question.
    doc_txt: Text content of the first retrieved document.
    res: Graded result from the retrieval grader.
    binary_score: Expected binary score should be 'yes'.
    """
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    """
    Tests the retrieval grader for a negative binary score.

    The function verifies that when invoking the retrieval grader
    with a question about making a pizza and a document content
    based on another question, the binary_score returned is "no".

    - question: The input question used to retrieve documents.
    - docs: List of documents retrieved based on the given question.
    - doc_txt: Page content of the first retrieved document.
    - res: Result of invoking the retrieval grader with a different
      question and the retrieved document content.
      It contains a binary_score that indicates if the document
      answers the question or not.

    Asserts:
      - Ensures that the binary_score is "no" indicating the
      document does not answer the question.
    """
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make a pizza", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    """

    Test the generation chain for a given question.

    This function retrieves documents based on the input question
    and then uses a generation mechanism to generate a response
    based on both the retrieved documents and the question.

    :param question: The input question to be processed.
    :type question: str
    :raises Exception: If either the retriever or generation chain fails.
    """
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    print(generation)


def test_hallucination_grader_answer_yes() -> None:
    """
    test_hallucination_grader_answer_yes
    Function to test the hallucination grader with an affirmative result.

    question: The question to be processed by the retriever.
    docs: Documents retrieved based on the provided question.

    generation: Response generated using the given documents and question.
    res: Result from the hallucination grader to check for hallucinations.

    assert: Checks if the binary score from the hallucination grader is true.
    """
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GraderHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    """
    Tests the hallucination grader for a response that should not be considered hallucinated.

    The test checks if the hallucination grader correctly identifies a given response
    as non-hallucinated based on the provided documents related to the question topic.
    Specifically, it ensures that the binary score returned by the hallucination grader
    is `False` for the given generation.

    Raises:
        AssertionError: If the hallucination grader incorrectly marks the response as hallucinated.
    """
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GraderHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make a pizza we need to first start with the dough.",
        }
    )
    assert not res.binary_score


def test_router_to_vectorstore() -> None:
    """
    Tests the routing functionality to the vector store.

    This function simulates querying a router with a specific question and
    asserts that the router correctly identifies the appropriate datasource
    as 'vectorstore'.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the router doesn't identify the datasource as 'vectorstore'
    """
    question = "agent memory"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    """
    Tests the routing functionality of the question_router to ensure it directs web searches correctly.

    It simulates a user query related to making a pizza and verifies that the router returns the expected datasource.

    Parameters: None

    Raises: AssertionError if the datasource returned by the router is not "websearch"
    """
    question = "how to make a pizza?"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"
