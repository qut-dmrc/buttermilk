import pytest

torch = pytest.importorskip("torch")
from buttermilk.toxicity.wildguard import Wildguard
from buttermilk.toxicity.types import EvalRecord

response_refuse = (False, """
        Harmful request: no
        Response refusal: yes
        Harmful response: no""")

response_harmful_request = (True, """
        Harmful request: yes
        Response refusal: no
        Harmful response: no""")

response_harmful_response = (True, """
        Harmful request: no
        Response refusal: no
        Harmful response: yes""")


class TestInterpreters:
    @pytest.mark.parametrize(
        ["expected", "response"],
        [response_refuse, response_harmful_request, response_harmful_response],
    )
    def test_wildguard(self, response, expected):
        result = Wildguard(client="dummy").interpret(response=response)
        assert isinstance(result, EvalRecord)
        assert int(result.predicted) == int(expected)
