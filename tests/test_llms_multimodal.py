import pytest
from langchain.prompts import ChatPromptTemplate
from rich import print as rprint

from buttermilk._core.runner_types import MediaObj, RecordInfo
from buttermilk.llms import MULTIMODAL_MODELS, LLMClient, LLMs


@pytest.mark.anyio
async def test_multimodal_question(
    llm: LLMClient,
    multimodal_record: RecordInfo,
):
    messages = []
    messages.append(("user", "Hi, can you please summarise this content for me?"))
    messages.append(
        multimodal_record.as_langchain_message(
            model_capabilities=llm.capabilities,
        ),
    )

    chain = ChatPromptTemplate.from_messages(messages) | llm.client
    response = await chain.ainvoke(input={})

    assert response


@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
def test_multimodal_input_b64_image(llms: LLMs, model, image_bytes):
    llm = llms[model]
    message = RecordInfo(
        data=[
            MediaObj(mime="image/jpg", content=image_bytes),
            "Hi, can you tell me what this is?",
        ],
    ).as_langchain_message(
        role="human",
        model_capabilities=llm.capabilities,
    )

    chain = ChatPromptTemplate.from_messages([message]) | llm.client
    answer = chain.invoke({})
    rprint(answer)
    assert answer


@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
def test_multimodal_input_b64_image_no_text(llms: LLMs, model, image_bytes):
    llm = llms[model]
    message = RecordInfo(
        data=[MediaObj(mime="image/jpg", content=image_bytes)],
    ).as_langchain_message(
        role="human",
        model_capabilities=llm.capabilities,
    )

    chain = ChatPromptTemplate.from_messages([message]) | llm.client
    answer = chain.invoke({})
    rprint(answer)
    assert answer


@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
def test_multimodal_input_video_uri(llms, model, video_url):
    llm = llms[model]
    message = RecordInfo(
        data=[
            MediaObj(mime="video/mp4", uri=video_url),
            "Hi, can you tell me what this is?",
        ],
    ).as_langchain_message(
        role="human",
        model_capabilities=llm.capabilities,
    )

    chain = ChatPromptTemplate.from_messages([message]) | llm.client
    answer = chain.invoke({})
    rprint(answer)
    assert answer


@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
def test_multimodal_input_b64_video(llms, model, video_bytes):
    llm = llms[model]
    message = RecordInfo(
        data=[
            MediaObj(mime="video/mp4", content=video_bytes),
        ],
    ).as_langchain_message(
        role="human",
        model_capabilities=llms.capabilities,
    )

    chain = ChatPromptTemplate.from_messages([message]) | llm.client
    answer = chain.invoke({})
    rprint(answer)
    assert answer
