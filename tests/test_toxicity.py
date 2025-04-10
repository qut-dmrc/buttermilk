# Let's provide an interface for all the various toxicity models

import pandas as pd
import pytest

pytest.importorskip("buttermilk.toxicity")
torch = pytest.importorskip("torch")
from buttermilk._core.types import Record
from buttermilk.toxicity import TOXCLIENTS
from buttermilk.toxicity.llamaguard import (
    LlamaGuard1Together,
    LlamaGuard2HF,
    LlamaGuard2Local,
    LlamaGuard2Replicate,
    LlamaGuard2Together,
    LlamaGuard3Local,
    LlamaGuard3LocalInt8,
    LlamaGuard3Together,
    LlamaGuardTox,
    MDJudgeLocal,
)

# from datatools.datapipes.toxic import (
#     BinaryHateSpeech,
#     Bold,
#     DragQueens,
#     Ethos,
#     HateCheckPipe,
#     HateMemes,
#     HateSpeechOffensive,
#     ImplicitHatePipe,
#     JigsawToxicComment,
#     RealToxicityPrompts,
#     ToxicChat,
#     Toxigen,
# )
from buttermilk.toxicity.toxicity import Comprehend, Perspective, ToxicChat
from buttermilk.toxicity.types import EvalRecord

LLAMAGUARD_MODELS = [
    LlamaGuard1Together,
    LlamaGuard2Replicate,
    LlamaGuard2Together,
    LlamaGuard2HF,
    LlamaGuard3Together,
]

LLAMAGUARD_GPU_MODELS = [
    LlamaGuard3Local,
    LlamaGuard3LocalInt8,
    LlamaGuard2Local,
]


@pytest.fixture(params=LLAMAGUARD_MODELS)
def llamaguard(request):
    return request.param()


@pytest.fixture(params=LLAMAGUARD_GPU_MODELS)
def llamaguard_gpu(request):
    return request.param()


@pytest.fixture(scope="session")
def toxic_record() -> Record:
    rec = Record(
        record_id="testdata_frenzal_rhomb",
        content="get fucked you fucking fuckwit you can't move into my house",
    )

    return rec


@pytest.fixture(scope="session", params=TOXCLIENTS)
def toxmodel(request):
    return request.param()


@pytest.fixture(scope="session")
def news_article():
    raise NotImplementedError


class TestLlamaGuard:
    @pytest.mark.parametrize(
        "standard",
        ["MDJUDGE.DOMAIN_POLICY", "MDJUDGE.TASK_POLICY"],
    )
    def test_prompt(self, standard):
        template = TEMPLATES.MDJUDGE.SALAD.fill(STANDARDS[standard])

        prompt = "[INST] " + template.to_text() + " [/INST]"
        prompt = prompt.format(prompt="TEST PROMPT")
        assert "TEST PROMPT" in prompt
        assert prompt.startswith(
            '[INST] Check if there is unsafe content in "Agent" response to the corresponding "User" instruction as shown in conversations according to our safety policy with below categories.',
        )

    def test_interpret(self, llamaguard: LlamaGuardTox):
        # check the right standard is being applied
        assert str.lower(type(llamaguard).__name__).startswith(llamaguard.standard)

        if llamaguard.standard == "llamaguard1":
            x = llamaguard.interpret("unsafe o3")
            assert x.scores[0].measure == "O3: Criminal Planning".upper()
        elif llamaguard.standard == "llamaguard2":
            x = llamaguard.interpret("unsafe s3")
            assert x.scores[0].measure == "S3: Sex Crimes".upper()
        elif llamaguard.standard == "llamaguard3":
            x = llamaguard.interpret("unsafe s3")
            assert x.scores[0].measure == "S3: Sex-Related Crimes".upper()

    @pytest.mark.skip(reason="Don't run local GPU tests")
    @pytest.mark.async_timeout(150)
    async def test_llamaguard_cpu(self, llamaguard_gpu):
        text = "The image is a black and white photo of a group of people in Ku Klux Klan (KKK) robes and hoods. There is text on the image that says 'i'm dreaming of a white christmas'."
        rec = Record(record_id="testHFModels", content=text)

        response = await llamaguard_gpu.mod_single(rec)
        assert response.result
        assert response.reasons is not None and "unsafe" in response.reasons

    async def test_llamaguard_defaultprompt(
        self,
        toxic_record: Record,
        llamaguard,
        logger,
    ):
        client = llamaguard()
        response = await client.mod_single(message=toxic_record)
        assert isinstance(response, EvalRecord)
        assert response.predicted is not None  # matches unsafe and safe. For now.
        assert response.reasons is not None and len(response.reasons) > 0

    #    @pytest.mark.skip(reason="Don't run local GPU tests")
    @pytest.mark.parametrize(
        "standard",
        ["MDJUDGE.DOMAIN_POLICY", "MDJUDGE.TASK_POLICY"],
    )
    async def test_mdjudge(self, toxic_record: Record, standard: str):
        client = MDJudgeLocal(standard=standard)
        response = await client.mod_single(message=toxic_record)
        assert isinstance(response, EvalRecord)
        assert response.predicted is not None  # matches unsafe and safe. For now.
        assert response.reasons is not None and len(response.reasons) > 0


class TestDataPipes:
    # @pytest.mark.parametrize(
    #     "pipe",
    #     [
    #         HateCheckPipe,
    #         ImplicitHatePipe,
    #         ToxicChat,
    #         RealToxicityPrompts,
    #         HateMemes,
    #         Bold,
    #         Toxigen,
    #         Ethos,
    #         BinaryHateSpeech,
    #         HateSpeechOffensive,
    #         DragQueens,
    #         JigsawToxicComment,
    #     ],
    # )
    def test_toxic_pipe(self, pipe):
        dp = pipe()
        for example in dp:
            assert isinstance(example, Record)
            assert example.source is not None and example.source != ""
            break

    def test_combined_text(self):
        # see if we can combine datasets together
        df = combined_text(group_sample_size=20)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 80


class TestClients:
    # These tests test the toxicity models directly
    def test_perspective(self, toxic_record: Record):
        client = Perspective()
        result = client.call_client(prompt=toxic_record.content)
        assert result is not None

        assert all(
            key in result.root
            for key in [
                "TOXICITY",
                "SEVERE_TOXICITY",
                "IDENTITY_ATTACK",
                "INSULT",
                "PROFANITY",
                "THREAT",
            ]
        )

    def test_comprehend(self, toxic_record: Record):
        tox = Comprehend()

        result = tox.client.mod_meratesingle(toxic_record.content)

        assert result is not None

        assert result.result_list, "result_list should not be empty"

        first_analysis_result = result.result_list[0]
        received_labels = {label.name for label in first_analysis_result.labels}

        expected_labels = {
            "PROFANITY",
            "HATE_SPEECH",
            "INSULT",
            "GRAPHIC",
            "HARASSMENT_OR_ABUSE",
            "SEXUAL",
            "VIOLENCE_OR_THREAT",
        }

        assert expected_labels <= received_labels, (
            f"Missing labels in the response; expected all of {expected_labels}"
        )

    #    @pytest.mark.skip(reason="Don't run local GPU tests")
    def test_REGARD(self, toxic_record: Record):
        # Test the REGARD wrapper
        from buttermilk.toxicity.regard import REGARD  # Import the wrapper

        client = REGARD()
        result = client.moderate(
            content=toxic_record.content, record_id=toxic_record.record_id
        )

        assert isinstance(result, EvalRecord)
        assert not result.error
        assert (
            result.predicted is not None
        )  # REGARD provides scores, predicted might be based on threshold
        assert any(
            s.measure == "NEGATIVE" for s in result.scores
        )  # Check for expected score measure

    @pytest.mark.parametrize("client_type", ["openai", "azure"])
    def test_OpenAIMod(self, client_type, toxic_record: Record):
        import openai

        openai.api_type = client_type
        client = openai.moderations
        result = client.create(input=toxic_record.content)
        assert isinstance(result, dict)

    async def test_ToxicChat(self, toxic_record):
        client = ToxicChat()
        async for result in client.moderate([toxic_record]):
            assert result is not None


class TestToxicityModels:
    """These tests test the toxicity models through the ToxicityModel wrapper"""

    @pytest.mark.parametrize("tox_model_cls", TOXCLIENTS)
    def test_mod(self, tox_model_cls, toxic_record: Record):
        tox_model = tox_model_cls()
        result = tox_model.moderate(
            content=toxic_record.content,
            record_id="test_record",
        )
        assert isinstance(result, EvalRecord)
        assert not result.error
        assert result.standard == tox_model.standard
        assert result.process == tox_model.process_chain
        assert result.model == tox_model.model
        assert all([s.measure for s in result.scores])
        assert result.predicted is not None

        # EvalSchema = read_yaml("datatools/chains/schemas/indicator.json")
        # uri = gc.upload_rows(
        #     EvalSchema, rows=rows, dataset="dmrc-analysis.tests.indicator"
        # )
        # assert uri

    # @pytest.mark.parametrize("tox_model_cls", TOXCLIENTS_LOCAL)
    # @pytest.mark.gpu
    # def test_tox_models_gpu(self, tox_model_cls, toxic_record):
    #     local_model = tox_model_cls()
    #     assert local_model
    #     result = local_model.moderate(content=toxic_record.content)
    #     assert isinstance(result, EvalRecord)
    #     assert not result.error
    #     assert result.standard == local_model.standard
    #     assert result.process == local_model.process_chain
    #     assert result.model == local_model.model
    #     assert all([s.measure for s in result.scores])
    #     assert result.predicted is not None


class TestIndicators:
    def test_bq_record(self, indicator):
        record = indicator.to_record()
        assert record


def test_moderate_success(mocker, toxmodel):
    # Mock the response from call_client
    mock_response = {"some": "response"}
    mock_call_client = mocker.patch.object(
        toxmodel,
        "call_client",
        return_value=mock_response,
    )

    # Mock the output from interpret
    mock_output = EvalRecord(record_id="record_id")
    mock_interpret = mocker.patch.object(
        toxmodel,  # Target the instance
        "interpret",
        return_value=mock_output,
    )

    # Call the method
    result = toxmodel.moderate(content="some text", record_id="record_id")

    # Assertions
    mock_call_client.assert_called_once_with("some text")
    mock_interpret.assert_called_once_with(mock_response)
    assert result == mock_output
    assert result.record_id == "record_id"  # Ensure record_id is passed


def test_moderate_interpret_error(mocker, toxmodel):
    # Mock the response from call_client
    mock_response = {"some": "response"}
    mock_call_client = mocker.patch.object(
        toxmodel,
        "call_client",
        return_value=mock_response,
    )

    # Mock interpret to raise a ValueError
    mock_interpret = mocker.patch.object(
        toxmodel,  # Target the instance
        "interpret",
        side_effect=ValueError("Interpretation error"),
    )

    # Call the method
    result = toxmodel.moderate(content="some text", record_id="record_id")

    # Assertions
    mock_call_client.assert_called_once_with("some text")
    mock_interpret.assert_called_once_with(mock_response)
    assert isinstance(result, EvalRecord)
    assert "Unable to interpret response" in result.error
    assert result.record_id == "record_id"  # Ensure record_id is passed
