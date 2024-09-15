from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
prompt = hub.pull("rlm/rag-prompt")

# generation_chain = prompt | llm | StrOutputParser()

from langchain_core.prompts import PromptTemplate
pirate_prompt_template = PromptTemplate.from_template(
    template="take {text} and answer like pirate"
)

generation_chain = (
    prompt | llm | StrOutputParser() | pirate_prompt_template | llm | StrOutputParser()
)
