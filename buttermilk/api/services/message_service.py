from typing import Any

from buttermilk._core.agent import AgentTrace
from buttermilk.agents.evaluators.scorer import QualResults
from buttermilk.agents.judge import JudgeReasons
from buttermilk.bm import logger


class MessageService:
    """Service for handling message processing with Pydantic objects directly"""

    @staticmethod
    def format_message_for_client(message: Any) -> dict[str, Any] | None:
        """Pass through the message directly to the client
        
        Args:
            message: The message to format (Pydantic object)
            
        Returns:
            dict[str, Any] | None: The serialized message or None if not serializable

        """
        # Skip message types that shouldn't be displayed in the UI
        if message is None:
            return None

        try:
            # Simply rely on the send_message method to properly serialize Pydantic objects
            return message
        except Exception as e:
            logger.error(f"Error formatting message for client: {e}")
            return None

    @staticmethod
    def extract_scores_from_messages(message_data_list: list[dict[str, Any]]):
        """Extract prediction and score data from message data entries.
        
        Args:
            message_data_list: List of message data entries (dicts with "message" key containing the original object)
            
        Returns:
            dict: Extracted scores and predictions keyed by agent name

        """
        results = {}

        # Process each message data entry
        for message_data in message_data_list:
            # Extract the actual message from the message data
            if "message" not in message_data:
                continue

            message = message_data["message"]

            # Process JudgeReasons entries
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

                # Use agent name as the dictionary key
                results[agent_name] = {
                    "call_id": call_id,
                    **prediction,
                }

        # Process scores in a second pass
        for message_data in message_data_list:
            if "message" not in message_data:
                continue

            message = message_data["message"]

            # Process QualResults entries
            if isinstance(message, AgentTrace) and isinstance(message.outputs, QualResults):
                # Find the matching prediction by call_id
                for agent_name, prediction_data in results.items():
                    if prediction_data["call_id"] == message.parent_call_id:
                        # Create the score object
                        score_obj = {
                            "assessor": message.outputs.assessor,
                            "assessments": [assessment.model_dump() for assessment in message.outputs.assessments],
                            "correctness": message.outputs.correctness,
                            "score_text": message.outputs.score_text,
                        }

                        # Create a safer approach without indexed assignments
                        # First, ensure there's a scores dictionary
                        if "scores" not in prediction_data:
                            scores = {}
                        else:
                            # Make a copy to avoid modifying the original
                            scores = dict(prediction_data["scores"])

                        # Update the copy with the new score
                        scores[message.call_id] = score_obj

                        # Update the prediction data with the modified scores dictionary
                        updated_prediction = dict(prediction_data)
                        updated_prediction["scores"] = scores

                        # Replace the entry in results
                        results[agent_name] = updated_prediction
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
