import re
from typing import Any, Dict, List, Optional, Tuple

from buttermilk.bm import logger


class UIService:
    """Service for UI-related utilities and message processing"""

    @staticmethod
    def extract_prediction_data(html_content: str, agent_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract prediction data from HTML content
        
        Args:
            html_content: The HTML content to extract prediction data from
            agent_info: Information about the agent
            
        Returns:
            Dict[str, Any]: The extracted prediction data, or None if extraction failed
        """
        try:
            role = agent_info.get("role", "").lower()
            
            # Basic prediction extraction using string matching
            prediction_data = {
                "agent_id": agent_info.get("id", ""),
                "agent_name": agent_info.get("name", role.capitalize()),
                "violates": "Violates: Yes" in html_content,
                "confidence": "medium",  # Default
                "conclusion": "",
                "reasons": [],
            }

            # Extract confidence
            if "Confidence: High" in html_content:
                prediction_data["confidence"] = "high"
            elif "Confidence: Medium" in html_content:
                prediction_data["confidence"] = "medium"
            elif "Confidence: Low" in html_content:
                prediction_data["confidence"] = "low"

            # Extract conclusion (simplified approach)
            if "<strong>Conclusion:</strong>" in html_content:
                conclusion_part = html_content.split("<strong>Conclusion:</strong>")[1].split("</div>")[0].strip()
                prediction_data["conclusion"] = conclusion_part

            # Extract reasons
            if "<strong>Reasoning:</strong>" in html_content:
                reasons_part = html_content.split("<strong>Reasoning:</strong>")[1]
                if "<li>" in reasons_part:
                    reason_items = reasons_part.split("<li>")[1:]
                    prediction_data["reasons"] = [r.split("</li>")[0].strip() for r in reason_items]

            # Return only if we have a conclusion
            if prediction_data["conclusion"]:
                return prediction_data
            
            return None
        except Exception as e:
            logger.warning(f"Error extracting prediction data: {e}")
            return None

    @staticmethod
    def extract_assessment_data(assessment_text: str) -> List[Dict[str, Any]]:
        """Extract assessment data from text
        
        Args:
            assessment_text: The text to extract assessment data from
            
        Returns:
            List[Dict[str, Any]]: The list of extracted assessments
        """
        assessments = []
        
        try:
            lines = assessment_text.split("\n")
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for correct/incorrect indicators
                correct = "✓" in line or "✅" in line or "correct" in line.lower()
                
                # Extract feedback text
                feedback = re.sub(r'^[✓✅✗❌]+\s*', '', line).strip()
                
                if feedback:
                    assessments.append({
                        "correct": correct,
                        "feedback": feedback
                    })
        except Exception as e:
            logger.warning(f"Error extracting assessment data: {e}")
        
        return assessments

    @staticmethod
    def format_error_message(error: Exception) -> str:
        """Format an error message for display
        
        Args:
            error: The exception to format
            
        Returns:
            str: The formatted error message
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        return f"""
        <div class="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-md">
            <div class="font-bold">{error_type} Error</div>
            <div class="text-sm mt-1">{error_message}</div>
        </div>
        """

    @staticmethod
    def format_loading_indicator(message: str = "Loading...") -> str:
        """Format a loading indicator
        
        Args:
            message: The loading message to display
            
        Returns:
            str: The formatted loading indicator
        """
        return f"""
        <div class="flex items-center justify-center p-4">
            <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500 mr-3"></div>
            <span>{message}</span>
        </div>
        """

    @staticmethod
    def format_agent_message(agent_type: str, name: str, content: str) -> str:
        """Format an agent message for display
        
        Args:
            agent_type: The type of agent (judge, scorer, assistant, etc.)
            name: The name of the agent
            content: The message content
            
        Returns:
            str: The formatted agent message
        """
        # Determine agent styling based on type
        background_class = "bg-gray-50"
        text_class = "text-gray-700"
        border_class = "border-gray-200"
        
        if agent_type == "judge":
            background_class = "bg-gray-100"
            text_class = "text-gray-700"
            border_class = "border-gray-300" 
        elif agent_type == "scorer":
            background_class = "bg-gray-50"
            text_class = "text-gray-900"
            border_class = "border-gray-400"
        elif agent_type == "assistant":
            background_class = "bg-blue-50"
            text_class = "text-blue-600"
            border_class = "border-blue-200"
        elif agent_type == "instructions":
            background_class = "bg-cyan-50"
            text_class = "text-cyan-600"
            border_class = "border-cyan-200"
        elif agent_type == "tool":
            background_class = "bg-green-50"
            text_class = "text-green-700"
            border_class = "border-green-200"
            
        return f"""
        <div class="p-3 rounded-lg my-2 {background_class} {text_class} border {border_class}">
            <div class="font-medium mb-1">{name}</div>
            <div class="prose prose-sm max-w-none">
                {content}
            </div>
        </div>
        """
