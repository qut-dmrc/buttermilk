import pytest
from langchain.prompts import ChatPromptTemplate

from buttermilk.llms import LLMs


@pytest.fixture
def llms() -> LLMs:
    return LLMs()


@pytest.fixture(params=AllModelNames)
def all_models(request):
    return request.param.name


def test_llm(llms, all_models):
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
