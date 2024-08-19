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

def tox_scorers(**kwargs) -> List[Dict[str, Union[str, int, float, list, Dict]]]:
    """This is a dummy function to generate a list of items.

    :param prefix: prefix to add to each item.
    :param size: number of items to generate.
    :param kwargs: other parameters.
    :return: a list of items. Each item is a dict with the following keys:
        - value: for backend use. Required.
        - display_value: for UI display. Optional.
        - hyperlink: external link. Optional.
        - description: information icon tip. Optional.
    """

    result = []

    for client in TOXCLIENTS:
        # Get defaults from each client (they probably should be class variables, but they're not...)
        params = {k:(v.default or '') for k,v in client.model_fields.items()}
        cur_item = {"value": client.__name__,
                    "display_value": params['model'],
                    "hyperlink": params['info_url'],
                    "description": f"Apply {params['standard']} standards to input using {params['model']} via {params['process_chain']}. {params['info_url']}".strip()
                    }
        result.append(cur_item)

    return result

input_text_dynamic_list_setting = DynamicList(function=tox_scorers)
input_settings = {
    "models": InputSetting(
        dynamic_list=input_text_dynamic_list_setting,
        allow_manual_entry=True,
        is_multi_select=True
    )}
@tool(
    name="Model Select",
    description="A list of available moderation and standards models",
    input_settings=input_settings
)
def moderation_models(models: list) -> str:
    return f"Hello {','.join(models)}"

if __name__ == '__main__':
    result = tox_scorers()
    print(result)