from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import PrivateAttr

BASE_DIR = Path(__file__).absolute().parent


import jmespath

from buttermilk.utils.templating import KeyValueCollector

"""
Allows YAML configuration for agents to aggregate their inputs
from other agents, defined parameters, and datasets.

Example YAML:
```yaml
    steps:
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
        params:
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


class FlowVariableRouter(KeyValueCollector):
    """Routes variables between workflow steps using mappings

    Data is essentially a dict of lists, where each key is the name of a step and
    each list is the output of an agent in a step.
    """

    _data: dict[str, Any ] = PrivateAttr(default_factory=dict)


    def _resolve_mappings(self, mappings: dict[str, Any]) -> dict[str, Any]:
        """Resolve all variable mappings to their values"""
        resolved = {}

        for target, source_spec in mappings.items():
            if isinstance(source_spec, Sequence) and not isinstance(source_spec, str):
                # Handle aggregation case
                resolved[target] = []
                for src in source_spec:
                    if isinstance(src, dict):
                        resolved[target].extend(self._resolve_mappings(src))
                    else:
                        resolved[target].extend(self._resolve_simple_path(src))
            elif isinstance(source_spec, Mapping|dict):
                # Handle nested mappings
                resolved[target] = self._resolve_mappings(source_spec)
            else:
                resolved[target] = self._resolve_simple_path(source_spec)

        # remove empty values
        resolved = {k: v for k, v in resolved.items() if v is not None and v != []}

        return resolved

    def _resolve_simple_path(self, path: str) -> Any:
        """Resolve a simple dot-notation path

        When a step has multiple outputs, returns a list with all matching results.
        For JMESPath expressions that return lists, flattens the results.
        """

        if "." not in path:
            # Direct reference to a step's complete output list
            return self._data.get(path, [])

        # Handle dot notation for nested fields
        step_name, field_path = path.split(".", 1)

        if step_name not in self._data:
            return None

        # Get all outputs for this step
        step_results = self._data[step_name]
        if not step_results:
            return None

        # Collect all matching results from all outputs
        all_matches = []

        for result in step_results:
            # Convert Pydantic models to dictionaries for JMESPath
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            else:
                result_dict = result

            value = jmespath.search(field_path, result_dict)
            if value is not None:
                # If the value is already a list, extend our results
                if isinstance(value, list):
                    all_matches.extend(value)
                else:
                    all_matches.append(value)

        # If no match was found, return None
        if not all_matches:
            return None

        return all_matches

