from typing import Any

from buttermilk._core.agent import AgentTrace
from buttermilk.agents.evaluators.scorer import QualResults
from buttermilk.agents.judge import JudgeReasons
from buttermilk.web.messages import _format_message_for_client as original_format_message


class MessageService:
    """Service for handling message formatting and processing with Pydantic objects"""

    @staticmethod
    def format_message_for_client(message: Any) -> dict[str, Any] | str | None:
        """Format a message for client display
        
        Args:
            message: The message to format (Pydantic object)
            
        Returns:
            Union[Dict[str, Any], str, None]: The formatted message

        """
        return original_format_message(message)

    @staticmethod
    def extract_scores_from_messages(messages: list[Any]):
        """Extract prediction and score data from messages.
        """
        results = {}
        for message in messages:
            if isinstance(message, AgentTrace) and isinstance(message.outputs, JudgeReasons):
                call_id = message.call_id
                agent_name = message.agent_info.name

                # Extract the actual fields needed by the template
                prediction = {
                    "agent": agent_name,
                    "prediction": message.outputs.prediction if hasattr(message.outputs, "prediction") else False,
                    "confidence": message.outputs.confidence if hasattr(message.outputs, "confidence") else None,
                    "conclusion": message.outputs.conclusion if hasattr(message.outputs, "conclusion") else "",
                    "reasons": message.outputs.reasons if hasattr(message.outputs, "reasons") else [],
                }

                # Use agent name as the dictionary key instead of call_id
                # But keep call_id as a reference
                results[agent_name] = {
                    "call_id": call_id,
                    **prediction,
                }

        # loop through again and attach scores this time
        for message in messages:
            if isinstance(message, AgentTrace) and isinstance(message.outputs, QualResults):
                # Find the matching prediction by call_id
                for agent_name, prediction_data in results.items():
                    if prediction_data["call_id"] == message.parent_call_id:
                        score = {
                            "assessor": message.outputs.assessor,
                            "assessments": [assessment.model_dump() for assessment in message.outputs.assessments],
                            "correctness": message.outputs.correctness,
                            "score_text": message.outputs.score_text,
                        }
                        results[agent_name]["scores"] = prediction_data.get("scores", {})
                        results[agent_name]["scores"][message.call_id] = score
                        break

        return results

    @staticmethod
    def get_pending_agents_from_progress(progress: dict[str, Any]) -> list[str]:
        """Get pending agents from progress data
        
        Args:
            progress: The progress data
            
        Returns:
            List[str]: The pending agents

        """
        if isinstance(progress, dict) and "pending_agents" in progress:
            return progress["pending_agents"]
        return []
