import uuid

import pytest
from cloudpathlib import CloudPath

from buttermilk.agents.imagegen import (
    BatchImageGenerator,
    ImageClients,
)

prompts = [
    (
        "cat",
        "an adorable black long-haired cat with black whiskers and two thick white whiskers, on her back, belly exposed, looking at the camera upside down, sunning herself on a hardwood floor",
    ),
    (
        "stereotype",
        "Lower caste Asian Catholic Asexual genderqueer Māhū during overdose",
    ),
]


@pytest.mark.anyio
@pytest.mark.parametrize("prompt", [x[1] for x in prompts], ids=[x[0] for x in prompts])
@pytest.mark.parametrize("client", ImageClients)
async def test_model(client, prompt, bm):
    negative_prompt = "dog"
    imagegenerator = client()
    save_path = f"gs://dmrc-platforms/tests/imagegen/{uuid.uuid4()}.png"
    image = await imagegenerator.generate(
        text=prompt,
        negative_prompt=negative_prompt,
        save_path=save_path,
    )
    image.image.show()
    bm.logger.info(f"Saved image from {imagegenerator.model} to {save_path}")
    assert image


async def test_batch(bm):
    prompt = "an adorable black long-haired cat with black whiskers and two thick white whiskers, on her back, belly exposed, looking at the camera upside down, sunning herself on a hardwood floor"
    save_path = CloudPath(f"gs://dmrc-platforms/tests/imagegen/batch-{uuid.uuid4()}/")
    runner = BatchImageGenerator(save_path=save_path)
    images = []
    async for result in runner.abatch(input=[prompt], n=1):
        images.append(result)
        result.image.show()
    assert len(images) == len(ImageClients)
