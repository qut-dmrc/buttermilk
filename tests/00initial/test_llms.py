from langchain.prompts import ChatPromptTemplate
import pytest

from buttermilk.llms import CHATMODELS, CHEAP_CHAT_MODELS

@pytest.fixture(scope="session")
def llms(bm):
    from buttermilk.llms import LLMs
    yield LLMs(connections=bm._connections_azure)

@pytest.mark.parametrize("cheapchatmodel", CHEAP_CHAT_MODELS)
def test_cheap_llm(llms, cheapchatmodel: str     ):
    llm = llms[cheapchatmodel]
    assert llm

    q = "hi! what's your name?"
    chain = ChatPromptTemplate.from_messages([("human", q)]) | llm
    answer = chain.invoke({})
    assert answer


@pytest.mark.parametrize("cheapchatmodel", CHATMODELS)
def test_all_llm(llms, cheapchatmodel: str     ):
    llm = llms[cheapchatmodel]
    assert llm

    q = "hi! what's your name?"
    chain = ChatPromptTemplate.from_messages([("human", q)]) | llm
    answer = chain.invoke({})
    assert answer
