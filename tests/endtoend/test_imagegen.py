import pytest

from buttermilk import logger

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


def pytest_generate_tests(metafunc):
    """Generate test parameters dynamically based on --run-expensive flag."""
    if "client" in metafunc.fixturenames:
        from buttermilk.agents.imagegen import ALL_IMAGE_CLIENTS, CHEAP_IMAGE_CLIENTS

        if metafunc.config.getoption("--run-expensive"):
            clients = ALL_IMAGE_CLIENTS
        else:
            clients = CHEAP_IMAGE_CLIENTS

        metafunc.parametrize("client", clients)


@pytest.mark.anyio
@pytest.mark.parametrize("prompt", [x[1] for x in prompts], ids=[x[0] for x in prompts])
async def test_model(client, real_bm, prompt):
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
