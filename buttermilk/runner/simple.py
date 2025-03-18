
from pydantic import BaseModel, PrivateAttr

from buttermilk._core.job import Job
from buttermilk._core.orchestrator import Orchestrator
from buttermilk._core.types import AgentInput
from autogen_core.model_context import (
    UnboundedChatCompletionContext,
)
from autogen_core.models import     AssistantMessage,UserMessage

class Sequencer(Orchestrator):
    _records: list = PrivateAttr(default_factory=list)
    _context: UnboundedChatCompletionContext = PrivateAttr(default_factory=UnboundedChatCompletionContext)

    async def run(self, request=None) -> None:
        prompt = await self.interface.get_input(message="Enter your prompt or record info...")
        msg = UserMessage(content=prompt, source="User")
        await self._context.add_message(msg)

        for step in self.agents:
            for variant in step:
                mapped_inputs = self._flow_data._resolve_mappings(variant.inputs)
            
                step_inputs = AgentInput(prompt=prompt, inputs=mapped_inputs, records=self._records)
                result = await variant(step_inputs)
                if not result.error:
                    self._flow_data.add(key=variant.name, value=result.response)
                    
                    # Harvest records
                    self._records.extend(result.records)

                    # also add to generic autogen collector
                    if result.response:
                        msg = AssistantMessage(content=result.response, source=variant.name)
                        await self._context.add_message(msg)
                    

                await self.interface.send_output(result)

            prompt = ''
            if not await self.interface.confirm("Proceed to next step?"):
                break
        return
    
    