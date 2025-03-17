from typing import Any

import regex as re
from autogen_core.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from promptflow.core._prompty_utils import parse_chat

from buttermilk._core.agent import AgentConfig
from buttermilk._core.runner_types import RecordInfo
from buttermilk.bm import BM, logger
from buttermilk.runner.chat import (
    Answer,
    BaseGroupChatAgent,
    NullAnswer,
    RequestToSpeak,
)
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.templating import (
    _parse_prompty,
    load_template,
)


class LLMAgent(BaseGroupChatAgent):
    def __init__(
        self,
        *,
        config: AgentConfig,
        group_chat_topic_type: str = "default",
        fail_on_unfilled_parameters: bool = False,
        description: str = "An agent that uses an LLM to respond to messages.",
    ) -> None:
        super().__init__(
            config=config,
            group_chat_topic_type=group_chat_topic_type,
        )
        bm = BM()
        self._json_parser = ChatParser()
        self._model_client = bm.llms.get_autogen_chat_client(self.parameters["model"])
        self._fail_on_unfilled_parameters = fail_on_unfilled_parameters

    async def fill_template(
        self,
        placeholder_messages: dict[
            str,
            list[SystemMessage | UserMessage | AssistantMessage],
        ] = {},
        untrusted_inputs: dict[str, Any] = {},
        context: list[SystemMessage | UserMessage | AssistantMessage] = [],
    ) -> list[Any]:
        """Fill the template with the given inputs and return a list of messages."""
        # Render the template using Jinja2
        rendered_template, unfilled_vars = load_template(
            parameters=self.parameters,
            untrusted_inputs=untrusted_inputs,
        )

        combined_placeholders = dict(context=context)
        combined_placeholders.update(placeholder_messages)

        # Interpret the template as a Prompty; split it into separate messages with
        # role and content keys. First we strip the header information from the markdown
        prompty = _parse_prompty(rendered_template)

        # Next we use Prompty's format to divide into messages and set roles
        messages = []
        for message in parse_chat(
            prompty,
            valid_roles=[
                "system",
                "user",
                "developer",
                "human",
                "placeholder",
                "assistant",
            ],
        ):
            # For placeholder messages, we are subbing in one or more
            # entire Message objects
            if message["role"] == "placeholder":
                # Remove everything except word chars to get the variable name
                var_name = re.sub(r"[^\w\d_]+", "", message["content"])
                try:
                    messages.extend(combined_placeholders[var_name])
                    # Remove the placeholder from the list of unfilled variables
                    if var_name in unfilled_vars:
                        unfilled_vars.remove(var_name)
                except KeyError as e:
                    err =  f"Missing {var_name} in template or placeholder vars.",
                    if self._fail_on_unfilled_parameters:
                        raise ValueError(err)
                    else: 
                        logger.warning(err)
                        
                continue

            # Check if there's content in the message after filling the template
            if re.sub(r"\s+", "", str(message["content"])):
                if message["role"] in ("system", "developer"):
                    messages.append(SystemMessage(content=message["content"]))
                elif message["role"] in ("assistant"):
                    messages.append(
                        AssistantMessage(
                            content=message["content"],
                            source=self.id.type,
                        ),
                    )
                else:
                    messages.append(
                        UserMessage(content=message["content"], source=self.id.type),
                    )

        if unfilled_vars:
            err = f"Template has unfilled parameters: {', '.join(unfilled_vars)}"
            if self._fail_on_unfilled_parameters:
                raise ValueError(err)
            else: 
                logger.warning(err)
                

        return messages

    async def query(
        self,
        request: RequestToSpeak,
    ) -> Answer | NullAnswer:

        # # Start by filling the history var, but overwrite if we've got another one passed in
        # history = "\n".join([ f"{msg.source}: {msg.content}" for msg in request.context if msg.content])
        # untrusted_vars = {
        #     "history": history,
        # }
        untrusted_vars= dict(**request.inputs)

        placeholders = {"context": request.context}
        placeholders= dict(**request.placeholders)
        # # Make sure there's a record too if possible
        # if 'record' not in request.placeholders and request.prompt or 'prompt' in request.inputs:
        #     from buttermilk.runner.moa import USER_AGENT_TYPE
        #     placeholders['record'] = UserMessage(content=request.prompt or request.inputs['prompt'], source=USER_AGENT_TYPE)
        messages = await self.fill_template(
            untrusted_inputs=untrusted_vars,
            placeholder_messages=placeholders,context=request.context,
        )

        response = await self._model_client.create(messages=messages)

        outputs = self._json_parser.parse(response.content)

        answer = Answer(
            agent_id=self.id.type,
            role="assistant",
            content=response.content,
            step=str(self.step),
            metadata=response.model_dump(exclude=["content"]),
            config=self.config,
            inputs=untrusted_vars,
            outputs=outputs,
            context=request.context,
        )
        return answer
