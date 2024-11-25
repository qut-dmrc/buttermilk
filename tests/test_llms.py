import pytest
from langchain.prompts import ChatPromptTemplate

from buttermilk.bm import BM
from buttermilk.llms import CHEAP_CHAT_MODELS, LLMs


@pytest.fixture(scope="session")
def llms(bm: BM):
    return bm.llms


@pytest.fixture
def all_models(llms: LLMs):
    for model in llms.model_names:
        yield model


@pytest.mark.parametrize("cheapchatmodel", CHEAP_CHAT_MODELS)
def test_cheap_llm(llms, cheapchatmodel: str):
    llm = llms[cheapchatmodel]
    assert llm

    q = "hi! what's your name?"
    chain = ChatPromptTemplate.from_messages([("human", q)]) | llm
    answer = chain.invoke({})
    assert answer


def test_all_llm(llms, all_models):
    llm = llms[all_models]
    assert llm

    q = "hi! what's your name?"
    chain = ChatPromptTemplate.from_messages([("human", q)]) | llm
    answer = chain.invoke({})
    assert answer


class TestPromptStyles:
    def test_words_in_mouth(self, cheapchatmodel, llms):
        llm = llms[cheapchatmodel]
        messages = [
            ("human", "hi! I'm Siobhan. What's your name?"),
            ("ai", "Hi Siobhan! I'm a chatbot, my developers call me"),
        ]
        chain = ChatPromptTemplate.from_messages(messages) | llm
        answer = chain.invoke({})
        assert answer
        assert answer.content.startswith(" ")  # starts with a space
        assert "Siobhan" not in answer.content
