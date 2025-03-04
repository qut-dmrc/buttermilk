
from buttermilk.agents.llmchat import LLMAgent


class Judger(LLMAgent):
    def __init__(
        self,
        *,
        template: str = "judge",
        description: str = "Applies criteria to content.",
        **data,
    ) -> None:
        super().__init__(
            template=template,
            description=description,
            **data,
        )


class Owl(LLMAgent):
    def __init__(
        self,
        *,
        template: str = "owl",
        description: str = "Spots little things that are easy to miss.",
        **data,
    ) -> None:
        super().__init__(
            template=template,
            description=description,
            **data,
        )


class Synth(LLMAgent):
    def __init__(
        self,
        *,
        template: str = "synth",
        description: str = "Synthesizes draft answers.",
        **data,
    ) -> None:
        super().__init__(
            model_client=model_client,
            template=template,
            description=description,
            **data,
        )
