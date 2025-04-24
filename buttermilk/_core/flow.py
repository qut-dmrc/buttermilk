from pathlib import Path

BASE_DIR = Path(__file__).absolute().parent




"""
Allows YAML configuration for agents to aggregate their inputs
from other agents, defined parameters, and datasets.

Example YAML:
```yaml
    agents:
    - judge: 
        # ...judge config...
    - draft:
        # ...draft config...
    - context:
        # ...context config...
    - synth:
        type: llm_agent
        model: gpt-4
        template: template_name  # no .jinja2 extension
        parameters:
            # Simple field access
            judge_answer: judge.answer
            
            # Nested structure
            analysis:
                scores: judge.scores[?score > `0.8`]
                feedback: draft.feedback
                context: context
            
            # Collecting multiple fields
            combined_answers:
            - judge.answer
            - draft.answer
            
            # answer: record   # a list of RecordInfo objects
            # fulltext: content  # a list of the text of all RecordInfo objects
            # prior_messages: context  # a list of messages received
            # history: history  # a list of the text of all messages received

            # Direct reference to whole step output
            feedback: context
            
            # Complex JMESPath expressions
            top_scores: "judge.scores[?score > `0.8`]"
            
            # Use existing variables
            job_id: job_id
```
"""
