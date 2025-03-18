
import asyncio

from autogen_core.model_context import (
    UnboundedChatCompletionContext,
)

from buttermilk._core.agent import Agent
from buttermilk.agents.llmchat import LLMAgent
from buttermilk.runner.chat import (
    FlowRequest,
    MessagesCollector,
)
from buttermilk.runner.moa import Conductor
from buttermilk.runner.varmap import FlowVariableRouter


class Selector(Conductor, LLMAgent):
    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        steps,
        fail_on_unfilled_parameters: bool = True,
    ):
        config = Agent(agent="LLMAgent", agent_id="conductor", description="description", parameters=dict(template="panel_host", model="gemini2pro", inputs={"participants": "participants", "context": "context", "history": "history", "prompt": "prompt", "record": "record"}))
        # Initialize LLMAgent first with its required parameters
        LLMAgent.__init__(
            self,
            config=config,
            group_chat_topic_type=group_chat_topic_type,
        )

        # Initialize Conductor attributes without calling its __init__
        # since we've already initialized the RoutedAgent base through LLMAgent
        self.description = description
        self._group_chat_topic_type = group_chat_topic_type
        self._placeholders = MessagesCollector()
        self._flow_data = FlowVariableRouter()
        self._context = UnboundedChatCompletionContext()
        self.running = False
        self.steps = steps

    async def run(self, init_text: str = None) -> None:
        # Dictionary to track agents by step
        step_agents = {}
        input_map = {}
        roles = []

        # Register all agent variants for each step
        for step_factory in self.steps:
            # save required inputs for each step
            input_map[step_factory.name] = step_factory.inputs

            # Register variants and collect the agent types
            agents_for_step = await step_factory.register_variants(
                self.runtime,
                group_chat_topic_type=self._group_chat_topic_type,
            )
            step_agents[step_factory.name] = agents_for_step

            # Add roles
            roles.append(dict(role=step_factory.name, description=step_factory.description))

        role_str = "\n".join([ f" - {str(role)}" for role in roles ])

        # Allow some time for initialization
        await asyncio.sleep(1)

        while await self.confirm_user():
            # Prompt the user for input
            prompt = await self.query_user(content="Enter your query")
            req = FlowRequest(inputs={"participants": role_str, "prompt": prompt}, context=await self._context.get_messages())
            response = await self.query(req)
            selected_topic_type = response.outputs.get("role",response.content)
            tasks = []

            if selected_topic_type in step_agents:
                # add the user's query if required
                step_data = self._flow_data._resolve_mappings(input_map[selected_topic_type])

                # And add our history too
                context = await self._context.get_messages()

                # Request each agent variant for this step to speak
                for agent_type in step_agents.get(selected_topic_type, []):
                    agent_id = await self.runtime.get(agent_type)
                    tasks.append(
                        self.runtime.send_message(
                            message=FlowRequest(
                                inputs=step_data,
                                placeholders=self._placeholders.get_dict(),
                                context=context,
                            ),
                            recipient=agent_id,
                        ),
                    )

                # Wait for all agents in this step to complete
                if tasks:
                    await asyncio.gather(*tasks)
                await asyncio.sleep(1)
            else:
                await self.publish(f"Invalid role selected: {response.content}")

