import trace
from promptflow.core import tool
from typing import List, Union, Dict
from promptflow.core import tool
from promptflow.connections import CustomStrongTypeConnection
from promptflow.connections import CustomConnection
from promptflow.contracts.types import Secret
from promptflow.core import ToolProvider, tool
from promptflow.tracing import trace
from dotenv import load_dotenv
from promptflow.entities import InputSetting, DynamicList

from buttermilk.toxicity.toxicity import *
from buttermilk.toxicity.toxicity import ToxicityModel, TOXCLIENTS

class Moderator(ToolProvider):
    """
    Doc reference :
    """

    def __init__(self, model: str):
        load_dotenv()
        self.model = model
        super().__init__()


    def __call__(self, content: str, **kwargs):
        return self.moderate(content=content, **kwargs)

    @tool
    def moderate(self, content: str, **kwargs) -> object:
        # TODO: can we put this here?
        # Get and instantiate the class from its string name
        cls = globals()[self.model]
        client: ToxicityModel = cls()
        response = client(content)

        return response

if __name__ == '__main__':
    moderator = Moderator(model="Perspective")
    result = moderator.moderate("This is a test input.")
    print(result)