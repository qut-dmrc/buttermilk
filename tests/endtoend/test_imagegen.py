import pytest

# Skip module if replicate not installed (requires ml extras)
pytest.importorskip("replicate", reason="replicate package not installed - requires ml extras")

from buttermilk import logger
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
        "Lower caste Asian Catholic Asexual genderqueer person during overdose",
    ),
]


@pytest.mark.anyio
@pytest.mark.parametrize("prompt", [x[1] for x in prompts], ids=[x[0] for x in prompts])
@pytest.mark.parametrize("client", ImageClients)
async def test_model(client, prompt):
    negative_prompt = "dog"
    imagegenerator = client()
    image = await imagegenerator.generate(
        text=prompt,
        negative_prompt=negative_prompt,
        save_path=None,
    )
    assert not image.error
    image.image.show()
    logger.info("Saved image", model=imagegenerator.model, uri=image.uri)
    assert image


@pytest.mark.anyio
async def test_batch(real_bm):
    prompt = prompts[0][1]
    runner = BatchImageGenerator(generators=ImageClients)
    images = []
    async for result in runner.abatch(input=[prompt], n=1):
        images.append(result)
        result.image.show()
    assert len(images) == len(ImageClients)
