"""
Enhanced Error Capture for Runtime Debugging

Flow-agnostic error capture and analysis system for buttermilk debugging.
Provides detailed error context, stack traces, and type checking diagnostics.
"""

import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from buttermilk._core.log import logger


class ErrorContext(BaseModel):
    """Structured error context information."""

    timestamp: str = Field(description="Error timestamp")
    error_type: str = Field(description="Error class name")
    error_message: str = Field(description="Error message")
    stack_trace: List[str] = Field(description="Full stack trace")
    function_name: str = Field(description="Function where error occurred")
    module_name: str = Field(description="Module where error occurred")
    line_number: int = Field(description="Line number where error occurred")
    local_variables: Dict[str, str] = Field(default_factory=dict, description="Local variables at error")
    type_info: Dict[str, Any] = Field(default_factory=dict, description="Type checking information")
    agent_context: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific context")


class TypeCheckingDiagnostics:
    """
    Diagnostics for type checking issues, especially subscripted generics errors.
    """

    @staticmethod
    def check_isinstance_calls(obj: Any, target_type: Any) -> Dict[str, Any]:
        """
        Safely check isinstance calls and diagnose type issues.
        
        Args:
            obj: Object to check
            target_type: Type to check against
            
        Returns:
            Diagnostic information about the type check
        """
        try:
            result = isinstance(obj, target_type)
            return {
                "success": True,
                "result": result,
                "obj_type": str(type(obj)),
                "target_type": str(target_type),
                "error": None
            }
        except TypeError as e:
            return {
                "success": False,
                "result": None,
                "obj_type": str(type(obj)),
                "target_type": str(target_type),
                "error": str(e),
                "is_subscripted_generic": "Subscripted generics" in str(e)
            }

    @staticmethod
    def fix_subscripted_isinstance(target_type: Any) -> Any:
        """
        Convert subscripted generic types to their base types for isinstance.
        
        Args:
            target_type: Potentially subscripted type
            
        Returns:
            Non-subscripted version safe for isinstance
        """
        import typing

        # Handle common subscripted generics
        origin = getattr(target_type, "__origin__", None)
        if origin is not None:
            return origin

        # Handle Union types
        if hasattr(typing, "get_origin") and typing.get_origin(target_type) is Union:
            args = typing.get_args(target_type)
            return tuple(TypeCheckingDiagnostics.fix_subscripted_isinstance(arg) for arg in args)

        # Return as-is if not subscripted
        return target_type

    @staticmethod
    def safe_isinstance(obj: Any, target_type: Any) -> bool:
        """
        Perform isinstance check safely, handling subscripted generics.
        
        Args:
            obj: Object to check
            target_type: Type to check against (may be subscripted)
            
        Returns:
            True if isinstance check succeeds, False otherwise
        """
        try:
            return isinstance(obj, target_type)
        except TypeError as e:
            if "Subscripted generics" in str(e):
                # Try with fixed type
                fixed_type = TypeCheckingDiagnostics.fix_subscripted_isinstance(target_type)
                try:
                    return isinstance(obj, fixed_type)
                except TypeError:
                    logger.warning(f"Could not fix subscripted generic: {target_type}")
                    return False
            else:
                raise


class ErrorCapture:
    """
    Flow-agnostic error capture system for debugging runtime issues.
    """

    def __init__(self, capture_locals: bool = True, max_local_length: int = 200):
        """
        Initialize error capture system.
        
        Args:
            capture_locals: Whether to capture local variables
            max_local_length: Maximum length for local variable values
        """
        self.capture_locals = capture_locals
        self.max_local_length = max_local_length
        self.captured_errors: List[ErrorContext] = []

    def capture_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """
        Capture detailed error information.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            
        Returns:
            Structured error context
        """
        tb = error.__traceback__
        if tb is None:
            tb = sys.exc_info()[2]

        # Get the last frame in the traceback
        while tb.tb_next is not None:
            tb = tb.tb_next

        frame = tb.tb_frame

        # Extract local variables if enabled
        local_vars = {}
        if self.capture_locals and frame.f_locals:
            for name, value in frame.f_locals.items():
                try:
                    str_value = str(value)
                    if len(str_value) > self.max_local_length:
                        str_value = str_value[:self.max_local_length] + "..."
                    local_vars[name] = str_value
                except Exception:
                    local_vars[name] = "<unprintable>"

        # Build error context
        error_context = ErrorContext(
            timestamp=datetime.now().isoformat(),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exception(type(error), error, error.__traceback__),
            function_name=frame.f_code.co_name,
            module_name=frame.f_globals.get("__name__", "unknown"),
            line_number=tb.tb_lineno,
            local_variables=local_vars,
            type_info=self._extract_type_info(error, frame),
            agent_context=context or {}
        )

        self.captured_errors.append(error_context)
        return error_context

    def _extract_type_info(self, error: Exception, frame) -> Dict[str, Any]:
        """Extract type-related information from error context."""
        type_info = {}

        # Check if this is a subscripted generics error
        if "Subscripted generics" in str(error):
            type_info["is_subscripted_generic_error"] = True
            type_info["error_details"] = str(error)

            # Try to identify the problematic isinstance call
            if frame.f_code:
                try:
                    # Get the source line if possible
                    import linecache
                    filename = frame.f_code.co_filename
                    lineno = frame.f_lineno
                    source_line = linecache.getline(filename, lineno).strip()
                    type_info["source_line"] = source_line

                    if "isinstance" in source_line:
                        type_info["contains_isinstance"] = True
                        type_info["source_context"] = {
                            "filename": filename,
                            "line_number": lineno,
                            "line_content": source_line
                        }
                except Exception:
                    pass

        return type_info

    @contextmanager
    def capture_context(self, context_name: str, **context_data):
        """
        Context manager for capturing errors with additional context.
        
        Args:
            context_name: Name for this context
            **context_data: Additional context data to capture
        """
        try:
            yield
        except Exception as e:
            context = {
                "context_name": context_name,
                **context_data
            }
            error_context = self.capture_error(e, context)
            logger.error(f"Error in {context_name}: {e}")
            logger.debug(f"Full error context: {error_context.model_dump()}")
            raise

    def wrap_function(self, func: Callable, context_data: Optional[Dict[str, Any]] = None):
        """
        Decorator to wrap functions with error capture.
        
        Args:
            func: Function to wrap
            context_data: Additional context data
            
        Returns:
            Wrapped function with error capture
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = {
                "function_name": func.__name__,
                "module_name": func.__module__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
                **(context_data or {})
            }

            with self.capture_context(f"function_{func.__name__}", **context):
                return func(*args, **kwargs)

        return wrapper

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of captured errors."""
        if not self.captured_errors:
            return {"total_errors": 0}

        error_types = {}
        subscripted_generic_errors = []

        for error in self.captured_errors:
            error_type = error.error_type
            error_types[error_type] = error_types.get(error_type, 0) + 1

            if error.type_info.get("is_subscripted_generic_error"):
                subscripted_generic_errors.append({
                    "timestamp": error.timestamp,
                    "function": error.function_name,
                    "module": error.module_name,
                    "line": error.line_number,
                    "message": error.error_message,
                    "source_line": error.type_info.get("source_line")
                })

        return {
            "total_errors": len(self.captured_errors),
            "error_types": error_types,
            "subscripted_generic_errors": subscripted_generic_errors,
            "latest_error": self.captured_errors[-1].model_dump() if self.captured_errors else None
        }


# Global error capture instance
_global_error_capture = ErrorCapture()


def capture_enhanced_rag_errors():
    """
    Decorator specifically for Enhanced RAG agent error capture.
    """
    def decorator(func):
        return _global_error_capture.wrap_function(
            func,
            context_data={"component": "enhanced_rag_agent"}
        )
    return decorator


def safe_isinstance_check(obj: Any, target_type: Any) -> bool:
    """
    Global safe isinstance check that handles subscripted generics.
    
    Args:
        obj: Object to check
        target_type: Type to check against
        
    Returns:
        True if isinstance check succeeds, False otherwise
    """
    return TypeCheckingDiagnostics.safe_isinstance(obj, target_type)


def analyze_type_checking_errors() -> Dict[str, Any]:
    """
    Analyze captured type checking errors and provide recommendations.
    
    Returns:
        Analysis and recommendations for fixing type errors
    """
    summary = _global_error_capture.get_error_summary()

    recommendations = []

    if summary.get("subscripted_generic_errors"):
        recommendations.append({
            "issue": "Subscripted generics in isinstance calls",
            "fix": "Replace isinstance(obj, List[str]) with isinstance(obj, list)",
            "affected_locations": [
                {
                    "function": err["function"],
                    "module": err["module"],
                    "line": err["line"],
                    "source": err.get("source_line")
                }
                for err in summary["subscripted_generic_errors"]
            ]
        })

    return {
        "summary": summary,
        "recommendations": recommendations,
        "total_type_errors": len(summary.get("subscripted_generic_errors", []))
    }


if __name__ == "__main__":
    # Quick analysis of current type checking errors
    print("🔍 TYPE CHECKING ERROR ANALYSIS")
    print("=" * 40)

    analysis = analyze_type_checking_errors()

    print(f"Total captured errors: {analysis['summary']['total_errors']}")
    print(f"Type checking errors: {analysis['total_type_errors']}")

    if analysis["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in analysis["recommendations"]:
            print(f"  - {rec['issue']}")
            print(f"    Fix: {rec['fix']}")
            for loc in rec["affected_locations"][:3]:  # Show first 3
                print(f"    Location: {loc['module']}.{loc['function']}:{loc['line']}")
                if loc.get("source"):
                    print(f"    Source: {loc['source']}")
    else:
        print("\n✅ No type checking errors captured yet")
