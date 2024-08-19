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

from datatools.chains.toxicity import *
from datatools.chains.toxicity import TOXCLIENTS


class Moderator(ToolProvider):
    """
    Doc reference :
    """

    def __init__(self, client_name: str):
        load_dotenv()
        self.client_name = client_name
        super().__init__()

        # TODO: can we put this here?
        # Get and instantiate the class from its string name
        cls = globals()[self.client_name]
        self.client: ToxicityModel = cls()

    @tool
    def moderate(self, input_text: str) -> object:
        response = self.client(input_text)

        return response

if __name__ == '__main__':
    moderator = Moderator(client_name="Perspective")
    result = moderator.moderate("This is a test input.")
    print(result)