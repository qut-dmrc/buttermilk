from buttermilk._core.contract import ConductorRequest, ManagerMessage, ManagerRequest, ManagerResponse, OOBMessages
from buttermilk.agents.llm import LLMAgent


class HostAgent(LLMAgent):
    """Coordinators for group chats that use an LLM."""

    async def handle_control_message(
        self,
        message: OOBMessages,
    ) -> None:
        # Respond to certain control messages addressed to us
        # for now drop everything though.
        return
