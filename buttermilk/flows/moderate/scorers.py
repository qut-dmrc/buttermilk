import trace
from promptflow.core import tool
from typing import List, Union, Dict
from promptflow.tracing import trace
from dotenv import load_dotenv

from buttermilk.toxicity.toxicity import *
from buttermilk.toxicity.toxicity import ToxicityModel

class Moderator():
    """
    Doc reference :
    """

    def __init__(self, model: str):
        load_dotenv()
        self.model = model
        super().__init__()

        # TODO: can we put this here?
        # Get and instantiate the class from its string name
        cls = globals()[self.model]
        self.client: ToxicityModel = cls()


    def __call__(self, content: str, **kwargs):
        return self.moderate(content=content, **kwargs)

    @trace
    def moderate(self, content: str, **kwargs) -> object:
        response = self.client(content)

        return response

if __name__ == '__main__':
    moderator = Moderator(model="Perspective")
    result = moderator.moderate("This is a test input.")
    print(result)