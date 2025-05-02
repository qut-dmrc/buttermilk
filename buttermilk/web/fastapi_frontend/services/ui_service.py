import re
from typing import Any, Dict, List, Optional

from buttermilk.agents.evaluators.scorer import QualResults, QualScoreCRA
from buttermilk.agents.judge import JudgeReasons
from buttermilk.bm import logger


class UIService:
    """Service for UI-related utilities and message formatting"""

    @staticmethod
    def format_judge_reasons_html(judge_reasons: JudgeReasons) -> str:
        """Format JudgeReasons as HTML for display
        
        Args:
            judge_reasons: The JudgeReasons object to format
            
        Returns:
            str: HTML representation of the judge reasons
        """
        if not judge_reasons:
            return ""
        
        # Access attributes directly from the Pydantic object
        conclusion = judge_reasons.conclusion
        prediction = judge_reasons.prediction
        confidence = judge_reasons.confidence.capitalize()
        reason_list = judge_reasons.reasons
        
        # Format prediction nicely
        prediction_color = "#28a745" if not prediction else "#dc3545"
        prediction_text = "No" if not prediction else "Yes"
        
        # Format confidence level
        confidence_color = "#28a745" if confidence == "High" else "#ffc107" if confidence == "Medium" else "#dc3545"
        
        # Create compact visual badges for violates/confidence
        prediction_bg = "rgba(0,0,0,0.1)"
        confidence_bg = "rgba(0,0,0,0.1)"
        
        badges_html = f"""
        <div style="display:flex; gap:8px; margin:3px 0 8px 0;">
            <span style="display:inline-flex; align-items:center; padding:2px 8px; background-color:{prediction_bg}; border:1px solid {prediction_color}; border-radius:12px; font-size:0.65rem;">
                <span style="font-weight:600;">Violates:</span> 
                <span style="margin-left:3px; font-weight:bold; color:{prediction_color};">{prediction_text}</span>
            </span>
            <span style="display:inline-flex; align-items:center; padding:2px 8px; background-color:{confidence_bg}; border:1px solid {confidence_color}; border-radius:12px; font-size:0.65rem;">
                <span style="font-weight:600;">Confidence:</span>
                <span style="margin-left:3px; font-weight:bold; color:{confidence_color};">{confidence}</span>
            </span>
        </div>
        """
        
        # Format reasons as a tooltip
        tooltip_html = ""
        if reason_list:
            reasons_items = [f"<li>{reason}</li>" for reason in reason_list]
            reasons_content = "\n".join(reasons_items)
            
            button_style = "background-color:#f8f9fa; border:1px solid #dee2e6; border-radius:12px; padding:2px 10px; font-size:0.7rem; display:inline-flex; align-items:center; margin-top:5px; color:#495057; cursor:pointer;"
            tooltip_style = "background-color:white; border:1px solid #dee2e6; border-radius:6px; box-shadow:0 2px 8px rgba(0,0,0,0.1); padding:10px; font-size:0.8rem; margin-top:8px; width:300px; position:absolute; z-index:10; transition:opacity 0.2s ease;"
            ul_style = "margin-top:6px; padding-left:20px;"
            
            tooltip_html = f"""
            <div class="group relative inline-block">
                <div class="cursor-pointer flex items-center">
                    <button style="{button_style}">
                        <span>View Detailed Reasoning</span>
                    </button>
                </div>
                <div class="invisible group-hover:visible absolute z-10 w-64 bg-white border border-gray-200 rounded-md shadow-lg p-3 text-sm mt-1 left-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300" style="{tooltip_style}">
                    <div>
                        <strong>Detailed Reasoning:</strong>
                        <ul style="{ul_style}">
                            {reasons_content}
                        </ul>
                    </div>
                </div>
            </div>
            """
        
        return f"""
        <div style="font-size:0.8rem;">
            <div style="margin-bottom:3px;"><strong>Conclusion:</strong> {conclusion}</div>
            {badges_html}
            {tooltip_html}
        </div>
        """

    @staticmethod
    def format_qual_results_html(qual_results: QualResults) -> str:
        """Format QualResults as HTML for display
        
        Args:
            qual_results: The QualResults object to format
            
        Returns:
            str: HTML representation of the qualitative results
        """
        if not qual_results:
            return ""
        
        # Get the score and appropriate color
        score = qual_results.correctness or 0.0
        score_color = UIService.get_score_color(score)
        score_text = qual_results.score_text
        
        # Format assessments
        assessments_html = ""
        if qual_results.assessments:
            assessment_items = []
            for assessment in qual_results.assessments:
                correct_icon = "✅" if assessment.correct else "❌"
                assessment_items.append(f"<li>{correct_icon} {assessment.feedback}</li>")
            
            assessments_html = f"""
            <div style="margin-top:10px;">
                <h3 style="font-size:0.9rem; font-weight:bold;">Assessment Details:</h3>
                <ul style="margin-top:5px; padding-left:20px;">
                    {"".join(assessment_items)}
                </ul>
            </div>
            """
        
        return f"""
        <div style="font-size:0.8rem; padding:10px; border-radius:5px; background:#f8f9fa; border:1px solid #dee2e6;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-weight:bold;">Score:</span> 
                    <span style="color:{score_color}; font-weight:bold;">{score_text}</span>
                </div>
                <div>
                    <span style="font-size:0.7rem; color:#666;">Assessor: {qual_results.assessor}</span>
                </div>
            </div>
            {assessments_html}
        </div>
        """

    @staticmethod
    def get_score_color(score: float) -> str:
        """Get appropriate color for a score value
        
        Args:
            score: The score value (0-1)
            
        Returns:
            str: Hex color code appropriate for the score
        """
        if score > 0.8:
            return "#28a745"  # Strong green
        if score > 0.6:
            return "#5cb85c"  # Light green
        if score > 0.4:
            return "#ffc107"  # Yellow
        if score > 0.2:
            return "#ff9800"  # Orange
        return "#dc3545"  # Red

    @staticmethod
    def format_error_message(error: Exception) -> str:
        """Format an error message for display
        
        Args:
            error: The exception to format
            
        Returns:
            str: The formatted error message as HTML
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
            str: The formatted loading indicator as HTML
        """
        return f"""
        <div class="flex items-center justify-center p-4">
            <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500 mr-3"></div>
            <span>{message}</span>
        </div>
        """

    @staticmethod
    def format_agent_message(agent_type: str, name: str, content: str) -> str:
        """Format an agent message for display using TailwindCSS classes
        
        Args:
            agent_type: The type of agent (judge, scorer, assistant, etc.)
            name: The name of the agent
            content: The message content
            
        Returns:
            str: The formatted agent message as HTML
        """
        # Map of agent types to styling classes
        agent_styles = {
            "judge": {
                "bg": "bg-gray-100", 
                "text": "text-gray-700", 
                "border": "border-gray-300"
            },
            "scorer": {
                "bg": "bg-gray-50", 
                "text": "text-gray-900", 
                "border": "border-gray-400"
            },
            "assistant": {
                "bg": "bg-blue-50", 
                "text": "text-blue-600", 
                "border": "border-blue-200"
            },
            "instructions": {
                "bg": "bg-cyan-50", 
                "text": "text-cyan-600", 
                "border": "border-cyan-200"
            },
            "tool": {
                "bg": "bg-green-50", 
                "text": "text-green-700", 
                "border": "border-green-200"
            },
            # Default styling
            "default": {
                "bg": "bg-gray-50", 
                "text": "text-gray-700", 
                "border": "border-gray-200"
            }
        }
        
        # Get styling for this agent type, or use default
        style = agent_styles.get(agent_type.lower(), agent_styles["default"])
            
        return f"""
        <div class="p-3 rounded-lg my-2 {style['bg']} {style['text']} border {style['border']}">
            <div class="font-medium mb-1">{name}</div>
            <div class="prose prose-sm max-w-none">
                {content}
            </div>
        </div>
        """
