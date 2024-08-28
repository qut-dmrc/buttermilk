# Let's provide an interface for all the various toxicity models

import json
import random

#import evaluate
import pandas as pd
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from buttermilk.toxicity.toxicity import *

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
from buttermilk.utils import read_yaml
from buttermilk.runner import InputRecord, Job, Consumer, ResultsCollector, TaskDistributor

LLAMAGUARD_MODELS = [
    LlamaGuard1Together,
    LlamaGuard2Replicate,
    LlamaGuard2Together,
    LlamaGuard2HF,
    LlamaGuard3HF,
    LlamaGuard3HFInt8,
    LlamaGuard3Together,
]

LLAMAGUARD_CPU_MODELS = [
    LlamaGuard3Local,
    LlamaGuard3LocalInt8,
    LlamaGuard2Local,
]

@pytest.fixture(params=LLAMAGUARD_MODELS)
def llamaguard(request):
    return request.param

@pytest.fixture(params=LLAMAGUARD_CPU_MODELS)
@pytest.mark.async_timeout(300)
def llamaguard_cpu(request):
    return request.param()

import pytest
from langchain_core.globals import set_debug

@pytest.fixture(scope="session")
def toxic_record() -> InputRecord:
    rec = InputRecord(
        record_id="You can't move into my house",
        text="get fucked you fucking fuckwit you can't move into my house",
        source="testdata_frenzal_rhomb",
    )

    return rec


@pytest.fixture(scope="session")
def news_article():
    raise NotImplementedError

class TestLlamaGuard:
    @pytest.mark.parametrize(
        "standard", ["MDJUDGE.DOMAIN_POLICY", "MDJUDGE.TASK_POLICY"]
    )
    def test_prompt(self, standard):
        template = TEMPLATES.MDJUDGE.SALAD.fill(STANDARDS[standard])

        prompt = "[INST] " + template.to_text() + " [/INST]"
        prompt = prompt.format(prompt="TEST PROMPT")
        assert "TEST PROMPT" in prompt
        assert prompt.startswith(
            '[INST] Check if there is unsafe content in "Agent" response to the corresponding "User" instruction as shown in conversations according to our safety policy with below categories.'
        )

    @pytest.mark.async_timeout(150)
    async def test_llamaguard_cpu(self, llamaguard_cpu):
        text = "The image is a black and white photo of a group of people in Ku Klux Klan (KKK) robes and hoods. There is text on the image that says 'i'm dreaming of a white christmas'."
        rec = InputRecord(record_id="testHFModels", text=text, source="testdata")

        response = await llamaguard_cpu.mod_single(rec)
        assert response.result == False
        assert response.reasons is not None and "unsafe" in response.reasons

    async def test_llamaguard_defaultprompt(
        self,
        toxic_record: InputRecord,
        llamaguard,logger
    ):
        client = llamaguard()
        response = await client.mod_single(message=toxic_record)
        assert isinstance(response, EvalRecord)
        assert response.predicted is not None  # matches unsafe and safe. For now.
        assert response.reasons is not None and len(response.reasons) > 0
        pass

    @pytest.mark.parametrize(
        "standard", ["MDJUDGE.DOMAIN_POLICY", "MDJUDGE.TASK_POLICY"]
    )
    async def test_mdjudge(self, toxic_record: InputRecord, standard: str):
        client = MDJudgeLocal(standard=standard)
        response = await client.mod_single(message=toxic_record)
        assert isinstance(response, EvalRecord)
        assert response.predicted is not None  # matches unsafe and safe. For now.
        assert response.reasons is not None and len(response.reasons) > 0
        pass

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
            assert isinstance(example, InputRecord)
            assert example.source is not None and example.source != ""
            break

    def test_combined_text(self):
        # see if we can combine datasets together
        df = combined_text(group_sample_size=20)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 80



class TestClients:

    # These tests test the toxicity models directly
    def test_perspective(self, toxic_record: InputRecord):
        client = Perspective()
        result = client.call_client(toxic_record.text)
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

    def test_comprehend(self, toxic_record: InputRecord):
        client = ComprehendAPI()
        result = client.mod_meratesingle(toxic_record.text)

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

        assert (
            expected_labels <= received_labels
        ), f"Missing labels in the response; expected all of {expected_labels}"

    def test_HONEST_prompt(self, toxic_record: InputRecord):
        # This isn't in the right format yet.
        # see https://huggingface.co/spaces/evaluate-measurement/honest

        honest = evaluate.load("honest", "en")
        completions = [toxic_record.text.split(" ")]
        result = honest.compute(predictions=completions)
        score = round(result["honest_score"], 3)
        assert score is not None
        assert isinstance(result, dict)

    def test_LFTW_R4(self, toxic_record: InputRecord):
        from transformers import pipeline

        pipe = pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
        )
        result = pipe(toxic_record.text)
        for r in result:
            assert r["label"] in ["hate", "nothate"]
            assert isinstance(r, dict)

    def test_REGARD(self, toxic_record: InputRecord):
        regard = evaluate.load("regard", module_type="measurement")
        result = regard.compute(data=toxic_record.text)["regard"][0]
        scores = {r["label"]: r["score"] for r in result}
        assert scores["negative"] > 0
        assert isinstance(scores, dict)

    @pytest.mark.parametrize("client_type", ["openai", "azure"])
    def test_OpenAIMod(self, client_type, toxic_record: InputRecord):
        import openai
        openai.api_type = client_type
        client = openai.moderations
        result = client.create(input=toxic_record.text)
        pass
        assert isinstance(result, dict)

    async def test_ToxicChat(self, toxic_record):
        client = ToxicChat()
        async for result in client.moderate([toxic_record]):
            assert result is not None
            pass


class TestToxicityModels:
    """These tests test the toxicity models through the ToxicityModel wrapper"""

    @pytest.fixture(
        scope="session",
        params=TOXCLIENTS,
    )
    def tox_model(self, request):
        yield request.param()

    @pytest.fixture(
        scope="session",
        params=TOXCLIENTS_LOCAL,
    )
    def local_model(self, request):
        yield request.param()

    @pytest.fixture()
    def example_with_answer(self) -> InputRecord:
        examples = _load_tests()
        example_category = random.choice(list(examples.values()))
        example = random.choice(example_category)
        return example

    def test_mod(self, tox_model: ToxicityModel, toxic_record: InputRecord):
        result = tox_model(toxic_record.text)
        assert isinstance(result, dict)
        assert not result.get('error')
        assert result['standard'] == tox_model.standard
        assert result['process'] == tox_model.process_chain
        assert result["model"] == tox_model.model
        assert all([s["measure"] for s in result["scores"]])
        assert result['predicted'] is not None


        # EvalSchema = read_yaml("datatools/chains/schemas/indicator.json")
        # uri = gc.upload_rows(
        #     EvalSchema, rows=rows, dataset="dmrc-analysis.tests.indicator"
        # )
        # assert uri

    def test_local_models(self, local_model, toxic_record):
        self.test_mod(tox_model=local_model, toxic_record=toxic_record)

class TestIndicators:
    def test_bq_record(self, indicator):
        record = indicator.to_record()
        assert record
        pass
