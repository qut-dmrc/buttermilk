import pytest

from buttermilk import logger
from buttermilk.agents.imagegen import (
    BatchImageGenerator,
    ImageClients,
)

# Skip entire module if replicate not installed (requires ml extras)
pytest.importorskip("replicate", reason="replicate package not installed - requires ml extras")

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
async def test_flux11pro_generation():
    """Test FLUX 1.1 Pro image generation via Azure."""
    from buttermilk.agents.imagegen import FLUX11Pro

    client = FLUX11Pro()
    image = await client.generate(
        text="a simple geometric shape on white background",
        save_path=None,
    )
    assert not image.error, f"Image generation failed: {image.error}"
    assert image.image is not None, "No image was generated"
    logger.info("FLUX 1.1 Pro image generated", uri=image.uri)


@pytest.mark.anyio
async def test_batch(real_bm):
    prompt = prompts[0][1]
    runner = BatchImageGenerator(generators=ImageClients)
    images = []
    async for result in runner.abatch(input=[prompt], n=1):
        images.append(result)
        result.image.show()
    assert len(images) == len(ImageClients)
