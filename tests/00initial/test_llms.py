import pytest
from langchain.prompts import ChatPromptTemplate

from buttermilk.bm import BM
from buttermilk.llms import CHEAP_CHAT_MODELS, LLMs


@pytest.fixture(scope="session")
def llms(bm: BM):
    return LLMs()


@pytest.fixture
def all_models(llms):
    for model in llms.model_names:
        yield llms[model]


@pytest.mark.parametrize("cheapchatmodel", CHEAP_CHAT_MODELS)
def test_cheap_llm(llms, cheapchatmodel: str):
    llm = llms[cheapchatmodel]
    assert llm

    q = "hi! what's your name?"
    chain = ChatPromptTemplate.from_messages([("human", q)]) | llm
    answer = chain.invoke({})
    assert answer


def test_all_llm(llms, all_models):
    llm = all_models
    assert llm

    q = "hi! what's your name?"
    chain = ChatPromptTemplate.from_messages([("human", q)]) | llm
    answer = chain.invoke({})
    assert answer
