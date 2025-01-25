import pytest
from langchain.prompts import ChatPromptTemplate

from buttermilk.llms import CHATMODELS, CHEAP_CHAT_MODELS


@pytest.mark.parametrize("model", CHATMODELS)
def test_all_llm(llms, model):
    llm = llms[model]
    assert llm

    q = "hi! what's your name?"
    chain = ChatPromptTemplate.from_messages([("human", q)]) | llm
    answer = chain.invoke({})
    assert answer


@pytest.mark.parametrize("cheapchatmodel", CHEAP_CHAT_MODELS)
def test_cheap_llm(llms, cheapchatmodel: str):
    llm = llms[cheapchatmodel]
    assert llm

    q = "hi! what's your name?"
    chain = ChatPromptTemplate.from_messages([("human", q)]) | llm
    answer = chain.invoke({})
    assert answer


class TestPromptStyles:
    @pytest.mark.parametrize("cheapchatmodel", CHEAP_CHAT_MODELS)
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
