from typing import Any
from urllib.error import HTTPError

import openai
from google.generativeai.types.generation_types import (
    BlockedPromptException,
    StopCandidateException,
)
from vertexai.generative_models._generative_models import (
    ResponseBlockedError,
    ResponseValidationError,
)

from buttermilk._core.exceptions import RateLimit

from .._core.log import getLogger

logger = getLogger()


########
#
# Extract error info from the variety of Exceptions we tend to hit. This is mainly
# to handle instances where we get blocked and want to know the reason.
#
########
def extract_error_info(e, process_info: dict = {}) -> dict[str, Any]:
    args = [str(x) for x in e.args]
    error_dict = dict(message=str(e), type=type(e).__name__, args=args)
    error_dict.update(process_info)
    try:
        error_dict["status_code"] = e.status_code
        error_dict["request_id"] = e.request_id
    except AttributeError:
        pass

    try:
        if isinstance(e, openai.APIStatusError):
            if e.status_code == 400:
                error_dict.update(
                    {
                        "error": "blocked",
                        "metadata": e.body.get("innererror", {}).get("content_filter_result", {}),
                        "code": e.body.get("innererror", {}).get("code"),
                    }
                )

            else:
                error_dict.update(e.body)
        elif isinstance(e, (IndexError, StopCandidateException)):
            # Gemini sometimes doesn't return a result?
            pass

        elif isinstance(e, ResponseBlockedError) or isinstance(e, ResponseValidationError):
            additional = try_extract_vertex_error(e)
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                raise RateLimit(str(e))

            error_dict.update({"error": "Prompt blocked by LLM", "error_info": additional})

        elif isinstance(e, BlockedPromptException):
            error_dict.update({"error": "Prompt blocked by LLM"})

        elif isinstance(e, TimeoutError):
            error_dict.update({"error": "Timeout error"})

        elif isinstance(e, ValueError):
            if e.args and "filter" in e.args[0]:
                error_dict.update({"error": "Prompt blocked by LLM"})

        elif isinstance(e, HTTPError) and (e.code in {400, 429}):
            raise RateLimit(*e.args)

        # Handle Google Vertex AI quota/rate limit errors
        elif hasattr(e, "reason") and ("quota" in str(e).lower() or "rate limit" in str(e).lower()):
            raise RateLimit(str(e))

    except Exception as secondary_error:
        logger.warning(
            f"Unable to extract error information from error: {type(e).__name__} {e} {e.args=}. Hit secondary error: {type(secondary_error).__name__} {secondary_error} {secondary_error.args=}"
        )

    return error_dict


def try_extract_vertex_error(e):
    info = []
    try:
        for resp in e.responses:
            try:
                info.append(resp.to_dict()["prompt_feedback"])
            except (KeyError, AttributeError) as err:
                logger.debug(f"Could not extract prompt_feedback from response: {err}")
            for cand in resp.candidates:
                candidate = {}
                try:
                    candidate["finish_reason"] = cand.finish_reason
                    candidate["partial"] = cand.parts.content
                except (AttributeError, IndexError) as err:
                    logger.debug(f"Could not extract candidate info: {err}")
                    continue
                if candidate:
                    info.append(candidate)
    except Exception as e:
        logger.debug(f"Failed to extract vertex error details: {type(e).__name__}: {e}")
        return ""
    return info
