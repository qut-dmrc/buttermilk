from typing import Any, Dict, List, Optional, Union

from buttermilk.bm import logger
from buttermilk.web.messages import _format_message_for_client as original_format_message
from buttermilk.web.fastapi_frontend.services.ui_service import UIService


class MessageService:
    """Service for handling message formatting and processing"""

    @staticmethod
    def format_message_for_client(message: Any) -> Optional[Union[Dict[str, Any], str]]:
        """Format a message for client display
        
        This is a wrapper around the original _format_message_for_client function from buttermilk.web.messages
        to maintain backward compatibility while adapting to the new architecture.
        
        Args:
            message: The message to format
            
        Returns:
            Union[Dict[str, Any], str, None]: The formatted message, which could be a dictionary for structured 
            messages, a string for HTML content, or None if the message shouldn't be displayed
        """
        return original_format_message(message)

    @staticmethod
    def extract_prediction_from_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract prediction data from a message
        
        Args:
            message: The message to extract prediction data from
            
        Returns:
            Dict[str, Any]: The extracted prediction data, or None if extraction failed
        """
        try:
            content = message.get("content", {})
            
            # Only process chat messages
            if not isinstance(content, dict) or content.get("type") != "chat_message":
                return None
                
            agent_info = content.get("agent_info", {})
            role = agent_info.get("role", "").lower()
            
            # Check if this is a judge or synthesizer message
            if role not in ["judge", "synthesiser"]:
                return None
                
            # Check if it contains reasoning
            html_content = content.get("content", "")
            if "JudgeReasons" not in str(html_content):
                return None
                
            return UIService.extract_prediction_data(html_content, agent_info)
                
        except Exception as e:
            logger.warning(f"Error extracting prediction from message: {e}")
            return None

    @staticmethod
    def extract_scores_from_messages(messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Extract score data from messages
        
        Args:
            messages: The messages to extract score data from
            
        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: A dictionary of scores, keyed by agent ID and then assessor ID
        """
        scores: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        for message in messages:
            content = message.get("content", {})
            
            # Process score updates
            if isinstance(content, dict) and content.get("type") == "score_update":
                score_data = content.get("score_data", {})
                agent_id = content.get("agent_id", "unknown")
                assessor_id = content.get("assessor_id", "scorer")
                
                if agent_id and assessor_id:
                    if agent_id not in scores:
                        scores[agent_id] = {}
                    scores[agent_id][assessor_id] = score_data
                    
        return scores

    @staticmethod
    def extract_predictions_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract prediction data from messages
        
        Args:
            messages: The messages to extract prediction data from
            
        Returns:
            List[Dict[str, Any]]: The list of extracted predictions
        """
        predictions = []
        
        for message in messages:
            prediction = MessageService.extract_prediction_from_message(message)
            if prediction:
                predictions.append(prediction)
                
        return predictions

    @staticmethod
    def get_pending_agents_from_progress(progress: Dict[str, Any]) -> List[str]:
        """Get pending agents from progress data
        
        Args:
            progress: The progress data
            
        Returns:
            List[str]: The list of pending agents
        """
        if isinstance(progress, dict) and "pending_agents" in progress:
            return progress["pending_agents"]
        return []
