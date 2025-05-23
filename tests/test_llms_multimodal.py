import pytest
from rich import print as rprint

from buttermilk._core.llms import MULTIMODAL_MODELS, LLMClient, LLMs
from buttermilk._core.types import MediaObj, Record


@pytest.mark.anyio
async def test_multimodal_question(
    llm: LLMClient,
    multimodal_record: Record,
):
    # messages.append(("user", "Hi, can you please summarise this content for me?"))
    # messages.append(
    #     multimodal_record.as_langchain_message(
    #         model_capabilities=llm.capabilities,
    #     ),
    # # )

    # chain = ChatPromptTemplate.from_messages(messages) | llm.client
    # response = await chain.ainvoke(input={})

    assert response


@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
def test_multimodal_input_b64_image(llms: LLMs, model, image_bytes):
    llms[model]
    Record(
        data=[
            MediaObj(mime="image/jpg", content=image_bytes),
            "Hi, can you tell me what this is?",
        ],
        # ).as_langchain_message(
        #     role="human",
        #     model_capabilities=llm.capabilities,
    )

    # chain = ChatPromptTemplate.from_messages([message]) | llm.client
    # answer = chain.invoke({})
    rprint(answer)
    assert answer
