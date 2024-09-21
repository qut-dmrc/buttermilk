
import json
import random
import pytest
from langchain_core.globals import set_debug

#import evaluate
import pandas as pd
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from buttermilk.flows.common.config import COL_PREDICTION
from buttermilk.toxicity import *
from buttermilk.toxicity import TOXCLIENTS
from buttermilk.toxicity.llamaguard import LlamaGuardTox
from buttermilk.toxicity import MDJudgeLocalTask,  MDJudgeLocalDomain


class TestSpecificExamples:

    @pytest.fixture(params=TOXCLIENTS)
    def toxclient(self, request: pytest.FixtureRequest):
        return request.param()

    @pytest.mark.parametrize("text", ["Fragile heterosexual males wouldn't be so frightened to be perceived as gay if they didn't treat gays like crap.", "Gay Penguin dads! https://cnn.com/2018/26/'"])
    def test_penguins(self, toxclient, text):
        response = toxclient.moderate(text)
        assert not response.error