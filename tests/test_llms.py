import pytest
from langchain.prompts import ChatPromptTemplate

from buttermilk._core.runner_types import MediaObj, RecordInfo
from buttermilk.bm import BM
from buttermilk.llms import CHATMODELS, CHEAP_CHAT_MODELS


@pytest.fixture(scope="session")
def llms(bm: BM):
    return bm.llms


@pytest.mark.parametrize("model", CHATMODELS)
def test_all_llm(llms, model):
    llm = llms[model]
    assert llm

    q = "hi! what's your name?"
    chain = ChatPromptTemplate.from_messages([("human", q)]) | llm
    answer = chain.invoke({})
    assert answer


@pytest.mark.parametrize("model", CHATMODELS)
def test_multimodal_input_b64(llms, model, image_bytes):
    llm = llms[model]
    message = RecordInfo(
        text="Hi, can you tell me what this is?",
        media=[MediaObj(mime="image/png", data=image_bytes)],
    ).as_openai_message(role="human")

    chain = ChatPromptTemplate.from_messages([message]) | llm
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
