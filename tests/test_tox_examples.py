import pytest

# import evaluate
pytest.importorskip("torch")
from buttermilk.toxicity import *
from buttermilk.toxicity import TOXCLIENTS


class TestSpecificExamples:
    @pytest.fixture(params=TOXCLIENTS)
    def toxclient(self, request: pytest.FixtureRequest):
        return request.param()

    @pytest.mark.parametrize(
        "text",
        [
            "Fragile heterosexual males wouldn't be so frightened to be perceived as gay if they didn't treat gays like crap.",
            "Gay Penguin dads! https://cnn.com/2018/26/'",
        ],
    )
    def test_penguins(self, toxclient, text):
        response = toxclient.moderate(text)
        assert not response.error
