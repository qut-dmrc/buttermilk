from typing import Any, Dict, List, Optional, Union

from buttermilk._core.agent import AgentTrace
from buttermilk._core.contract import (
    ConductorResponse,
    ManagerRequest,
    TaskProcessingComplete,
    TaskProcessingStarted,
    TaskProgressUpdate,
    ToolOutput,
)
from buttermilk._core.types import Record
from buttermilk.agents.evaluators.scorer import QualResults
from buttermilk.agents.judge import JudgeReasons
from buttermilk.bm import logger
from buttermilk.web.messages import _format_message_for_client as original_format_message


class MessageService:
    """Service for handling message formatting and processing with Pydantic objects"""

    @staticmethod
    def format_message_for_client(message: Any) -> Optional[Union[Dict[str, Any], str]]:
        """Format a message for client display
        
        Args:
            message: The message to format (Pydantic object)
            
        Returns:
            Union[Dict[str, Any], str, None]: The formatted message
        """
        return original_format_message(message)

    @staticmethod
    def get_judge_reasons(message: Any) -> Optional[JudgeReasons]:
        """Get JudgeReasons directly from a message if available
        
        Args:
            message: The message (typically an AgentTrace)
            
        Returns:
            Optional[JudgeReasons]: The JudgeReasons object if available
        """
        if isinstance(message, AgentTrace) and isinstance(message.outputs, JudgeReasons):
            return message.outputs
        return None

    @staticmethod
    def get_qual_results(message: Any) -> Optional[QualResults]:
        """Get QualResults directly from a message if available
        
        Args:
            message: The message (typically an AgentTrace)
            
        Returns:
            Optional[QualResults]: The QualResults object if available
        """
        if isinstance(message, AgentTrace) and isinstance(message.outputs, QualResults):
            return message.outputs
        return None

    @staticmethod
    def extract_scores_from_messages(messages: List[Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Extract score data from messages with direct access to QualResults objects
        
        Args:
            messages: The original messages
            
        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: Scores by agent ID and assessor ID
        """
        scores: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        for message in messages:
            qual_results = MessageService.get_qual_results(message)
            if qual_results:
                agent_id = qual_results.agent_id
                assessor_id = qual_results.assessor
                
                if agent_id and assessor_id:
                    if agent_id not in scores:
                        scores[agent_id] = {}
                        
                    scores[agent_id][assessor_id] = {
                        "correctness": qual_results.correctness,
                        "score_text": qual_results.score_text,
                        "assessments": [assessment.model_dump() for assessment in qual_results.assessments]
                    }
                    
        return scores

    @staticmethod
    def extract_predictions_from_messages(messages: List[Any]) -> List[JudgeReasons]:
        """Extract JudgeReasons objects directly from messages
        
        Args:
            messages: The original messages
            
        Returns:
            List[JudgeReasons]: The extracted JudgeReasons objects
        """
        predictions: List[JudgeReasons] = []
        
        for message in messages:
            judge_reasons = MessageService.get_judge_reasons(message)
            if judge_reasons:
                predictions.append(judge_reasons)
                
        return predictions

    @staticmethod
    def get_pending_agents_from_progress(progress: Dict[str, Any]) -> List[str]:
        """Get pending agents from progress data
        
        Args:
            progress: The progress data
            
        Returns:
            List[str]: The pending agents
        """
        if isinstance(progress, dict) and "pending_agents" in progress:
            return progress["pending_agents"]
        return []
