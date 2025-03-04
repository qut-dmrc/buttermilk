from buttermilk.agents.llmchat import LLMAgent


class Judger(LLMAgent):
    template: str = "judge"
    description: str = "Applies criteria to content."


class Owl(LLMAgent):
    template: str = "owl"
    description: str = "Spots little things that are easy to miss."


class Synth(LLMAgent):
    template: str = "synth"
    description: str = "Synthesizes draft answers."
