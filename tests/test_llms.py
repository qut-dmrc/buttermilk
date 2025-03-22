import pytest

from buttermilk._core.runner_types import Record
from buttermilk.llms import CHATMODELS, CHEAP_CHAT_MODELS, LLMClient


@pytest.mark.parametrize("model", CHATMODELS)
def test_all_llm(llms, model):
    llm = llms[model]
    assert llm

    q = "hi! what's your name?"
    assert answer


@pytest.mark.parametrize("cheapchatmodel", CHEAP_CHAT_MODELS)
def test_cheap_llm(llms, cheapchatmodel: str):
    llm = llms[cheapchatmodel]
    assert llm

    q = "hi! what's your name?"


@pytest.mark.anyio
async def test_text_question(
    llm: LLMClient,
    text_record: Record,
):
    messages = []
    messages.append(("user", "Hi, can you please summarise this content for me?"))
    # messages.append(
    #     text_record.as_langchain_message(
    #         model_capabilities=llm.capabilities,
    #     ),
    # )

    chain = ChatPromptTemplate.from_messages(messages) | llm.client
    response = await chain.ainvoke(input={})

    assert response


class TestPromptStyles:
    @pytest.mark.parametrize("cheapchatmodel", CHEAP_CHAT_MODELS)
    def test_words_in_mouth(self, cheapchatmodel, llms):
        llm = llms[cheapchatmodel]
        messages = [
            ("human", "hi! I'm Siobhan. What's your name?"),
            ("ai", "Hi Siobhan! I'm a chatbot, my developers call me"),
        ]
        chain = ChatPromptTemplate.from_messages(messages) | llm.client
        answer = chain.invoke({})
        assert answer
        assert answer.content.startswith(" ")  # starts with a space
        assert "Siobhan" not in answer.content
