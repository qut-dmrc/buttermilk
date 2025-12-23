"""Session storage service for persisting chat flow messages to disk."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import List, Optional

from buttermilk import bm, logger
from buttermilk.api.services.message_service import ChatMessage
from buttermilk.utils import scrub_serializable

# Default sessions directory - will be overridden by get_sessions_dir()
SESSIONS_DIR = Path("data/sessions")


def get_sessions_dir() -> Path:
    """Get the configured sessions directory from BM instance.

    Returns:
        Path to the sessions directory from bm.session_info.sessions_dir,
        falling back to SESSIONS_DIR if BM is not available.
    """
    try:
        if bm and bm.session_info and hasattr(bm.session_info, "sessions_dir"):
            return Path(bm.session_info.sessions_dir)
    except Exception:
        # Fall back to default if BM is not available or configured
        pass
    return SESSIONS_DIR


class SessionStorageService:
    """Service for persisting and retrieving session messages.

    This service handles:
    - Saving messages to JSON files organized by session ID
    - Loading historical messages for session restoration
    - Filtering out non-substantive system messages
    - Managing session file lifecycle
    """

    def __init__(self, sessions_dir: Optional[Path] = None):
        """Initialize the session storage service.

        Args:
            sessions_dir: Optional custom directory for session files.
                         Defaults to bm.session_info.sessions_dir or data/sessions/
        """
        self.sessions_dir = sessions_dir or get_sessions_dir()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session.

        Args:
            session_id: The session identifier

        Returns:
            Path to the session JSONL file
        """
        return self.sessions_dir / f"{session_id}.jsonl"

    def _append_entry(self, session_id: str, entry: dict) -> None:
        """Append a single entry to the session JSONL file.

        Args:
            session_id: The session identifier
            entry: Dictionary entry to append

        Raises:
            IOError: If file cannot be written
        """
        session_file = self._get_session_file(session_id)

        try:
            with open(session_file, "a", encoding="utf-8") as f:
                json.dump(entry, f)
                f.write("\n")
        except IOError as e:
            logger.error(
                "Failed to append entry to session file",
                session_id=session_id,
                error=e,
            )
            raise

    def _get_or_create_session_data(self, session_id: str) -> dict:
        """Get existing session data by reading JSONL file.

        Reconstructs session data structure from JSONL entries.
        Each entry has a _type field indicating its type (message, parameter_update, etc.).

        Args:
            session_id: The session identifier

        Returns:
            Dictionary containing session data reconstructed from JSONL entries

        Raises:
            json.JSONDecodeError: If JSONL entries are malformed
        """
        session_file = self._get_session_file(session_id)

        # Create new session data if file doesn't exist
        if not session_file.exists():
            return self._create_new_session_data(session_id)

        # Reconstruct session data from JSONL entries
        session_data = self._create_new_session_data(session_id)
        messages = []

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                        entry_type = entry.get("_type")

                        if entry_type == "message":
                            # Remove metadata fields before storing
                            entry.pop("_type", None)
                            entry.pop("_timestamp", None)
                            messages.append(entry)
                        elif entry_type == "parameter_update":
                            session_data["parameters"] = entry.get("parameters", {})
                            session_data["last_updated"] = entry.get("_timestamp")
                            session_data["last_activity"] = entry.get("_timestamp")
                        elif entry_type == "status_update":
                            session_data["flow_status"] = entry.get("status", "idle")
                            session_data["last_updated"] = entry.get("_timestamp")
                            session_data["last_activity"] = entry.get("_timestamp")
                        elif entry_type == "completion":
                            session_data["completed_at"] = entry.get("completed_at")
                            session_data["last_updated"] = entry.get("_timestamp")
                            session_data["last_activity"] = entry.get("_timestamp")
                        elif entry_type == "session_init":
                            session_data["created_at"] = entry.get("_timestamp")
                            session_data["last_updated"] = entry.get("_timestamp")
                            session_data["last_activity"] = entry.get("_timestamp")

                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Skipping malformed JSONL entry",
                            session_file=session_file,
                            line_num=line_num,
                            error=e,
                        )
                        continue

            session_data["messages"] = messages
            return session_data

        except Exception as e:
            logger.error(
                "Failed to read session file",
                session_file=session_file,
                error=e,
            )
            raise

    def save_message(self, session_id: str, message: ChatMessage) -> None:
        """Save a message to the session file using incremental append.

        Messages are appended as JSONL entries to the session file.
        Each entry includes _type: "message" and _timestamp for tracking.

        Args:
            session_id: The session identifier
            message: The ChatMessage to persist

        Raises:
            IOError: If file cannot be written
        """
        if not self.should_persist_message(message):
            logger.debug("Skipping persistence for message", message_type=message.type)
            return

        try:
            # Convert message to dict and add metadata
            message_dict = scrub_serializable(message.model_dump())
            message_dict["_type"] = "message"
            message_dict["_timestamp"] = datetime.now(UTC).isoformat()

            # Append to file
            self._append_entry(session_id, message_dict)

            logger.debug(
                "Saved message to session",
                message_id=message.message_id,
                session_id=session_id,
            )

        except Exception as e:
            logger.error(
                "Failed to save message to session", session_id=session_id, error=e
            )
            raise

    def save_parameters(self, session_id: str, parameters: dict) -> None:
        """Save flow parameters to the session file using incremental append.

        Args:
            session_id: The session identifier
            parameters: Dictionary containing flow parameters (flow, record_id, criteria, etc.)

        Raises:
            IOError: If file cannot be written
        """
        try:
            entry = {
                "_type": "parameter_update",
                "_timestamp": datetime.now(UTC).isoformat(),
                "parameters": parameters,
            }

            self._append_entry(session_id, entry)

            logger.debug(
                "Saved parameters for session",
                parameters=parameters,
                session_id=session_id,
            )

        except Exception as e:
            logger.error(
                "Failed to save parameters for session", session_id=session_id, error=e
            )
            raise

    def update_flow_status(self, session_id: str, status: str) -> None:
        """Update the flow status for a session using incremental append.

        Args:
            session_id: The session identifier
            status: New flow status (idle, running, completed, failed)

        Raises:
            IOError: If file cannot be written
        """
        try:
            entry = {
                "_type": "status_update",
                "_timestamp": datetime.now(UTC).isoformat(),
                "status": status,
            }

            self._append_entry(session_id, entry)

            logger.debug("Updated flow status", status=status, session_id=session_id)

        except Exception as e:
            logger.error(
                "Failed to update flow status for session",
                session_id=session_id,
                error=e,
            )
            raise

    def get_flow_status(self, session_id: str) -> str:
        """Get the current flow status for a session.

        Args:
            session_id: The session identifier

        Returns:
            Current flow status or 'idle' if session doesn't exist
        """
        session_file = self._get_session_file(session_id)

        if not session_file.exists():
            return "idle"

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            return session_data.get("flow_status", "idle")

        except Exception as e:
            logger.error(
                "Failed to read flow status for session", session_id=session_id, error=e
            )
            return "idle"

    def get_session_parameters(self, session_id: str) -> dict:
        """Get the flow parameters for a session.

        Args:
            session_id: The session identifier

        Returns:
            Dictionary containing flow parameters or empty dict if session doesn't exist
        """
        session_file = self._get_session_file(session_id)

        if not session_file.exists():
            return {}

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            return session_data.get("parameters", {})

        except Exception as e:
            logger.error(
                "Failed to read parameters for session", session_id=session_id, error=e
            )
            return {}

    def is_session_stale(self, session_id: str, stale_minutes: int = 30) -> bool:
        """Check if a session is stale based on last activity.

        Args:
            session_id: The session identifier
            stale_minutes: Minutes of inactivity to consider stale (default: 30)

        Returns:
            True if session is stale or doesn't exist, False otherwise
        """
        session_file = self._get_session_file(session_id)

        if not session_file.exists():
            return True

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            last_activity_str = session_data.get("last_activity")
            if not last_activity_str:
                return True

            last_activity = datetime.fromisoformat(last_activity_str)
            stale_threshold = datetime.now(UTC) - timedelta(minutes=stale_minutes)

            return last_activity < stale_threshold

        except Exception as e:
            logger.error(
                "Failed to check staleness for session", session_id=session_id, error=e
            )
            return True

    def get_session_messages(self, session_id: str) -> List[ChatMessage]:
        """Retrieve all messages for a session from JSONL file.

        Args:
            session_id: The session identifier

        Returns:
            List of ChatMessage objects, empty list if session doesn't exist
            or file is corrupted
        """
        session_file = self._get_session_file(session_id)

        if not session_file.exists():
            logger.debug("Session file not found", session_id=session_id)
            return []

        messages = []

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)

                        # Only process message entries
                        if entry.get("_type") != "message":
                            continue

                        # Remove metadata fields before parsing as ChatMessage
                        entry.pop("_type", None)
                        entry.pop("_timestamp", None)

                        message = ChatMessage(**entry)
                        messages.append(message)

                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Skipping malformed JSONL entry",
                            session_file=session_file,
                            line_num=line_num,
                            error=e,
                        )
                        continue
                    except Exception as e:
                        logger.warning(
                            "Failed to parse message in session",
                            session_id=session_id,
                            line_num=line_num,
                            error=e,
                        )
                        continue

            logger.info(
                "Retrieved messages for session",
                message_count=len(messages),
                session_id=session_id,
            )
            return messages

        except Exception as e:
            logger.error("Failed to read session", session_id=session_id, error=e)
            return []

    def session_exists(self, session_id: str) -> bool:
        """Check if a session file exists.

        Args:
            session_id: The session identifier

        Returns:
            True if the session file exists, False otherwise
        """
        return self._get_session_file(session_id).exists()

    def should_persist_message(self, message: ChatMessage) -> bool:
        """Determine if a message should be persisted.

        Filters out system status updates and other non-substantive messages
        to keep session files focused on actual conversation content.

        Args:
            message: The ChatMessage to evaluate

        Returns:
            True if the message should be saved, False otherwise
        """
        # Message types to exclude from persistence
        exclude_types = {"system_update", "system_message"}

        return message.type not in exclude_types

    def _create_new_session_data(self, session_id: str) -> dict:
        """Create a new session data structure.

        Args:
            session_id: The session identifier

        Returns:
            Dictionary with session metadata and empty messages list
        """
        return {
            "session_id": session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "last_updated": datetime.now(UTC).isoformat(),
            "flow_status": "idle",  # idle, running, completed, failed
            "last_activity": datetime.now(UTC).isoformat(),
            "parameters": {},  # flow parameters (flow, record_id, criteria, etc.)
            "messages": [],
        }

    def delete_session(self, session_id: str) -> bool:
        """Delete a session file.

        Args:
            session_id: The session identifier

        Returns:
            True if deletion was successful, False otherwise
        """
        session_file = self._get_session_file(session_id)

        if not session_file.exists():
            logger.debug("Session file not found for deletion", session_id=session_id)
            return False

        try:
            session_file.unlink()
            logger.info("Deleted session file", session_id=session_id)
            return True
        except Exception as e:
            logger.error("Failed to delete session", session_id=session_id, error=e)
            return False

    def get_session_metadata(self, session_id: str) -> Optional[dict]:
        """Get session metadata without loading all messages.

        Args:
            session_id: The session identifier

        Returns:
            Dictionary with session metadata (created_at, last_updated, message_count)
            or None if session doesn't exist
        """
        session_file = self._get_session_file(session_id)

        if not session_file.exists():
            return None

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            return {
                "session_id": session_data.get("session_id"),
                "created_at": session_data.get("created_at"),
                "last_updated": session_data.get("last_updated"),
                "flow_status": session_data.get("flow_status", "idle"),
                "last_activity": session_data.get("last_activity"),
                "message_count": len(session_data.get("messages", [])),
            }
        except Exception as e:
            logger.error(
                "Failed to read session metadata", session_id=session_id, error=e
            )
            return None

    def list_sessions(self) -> List[dict]:
        """List all available sessions with metadata.

        Returns:
            List of session metadata dictionaries
        """
        sessions = []

        for session_file in self.sessions_dir.glob("*.jsonl"):
            session_id = session_file.stem
            metadata = self.get_session_metadata(session_id)
            if metadata:
                sessions.append(metadata)

        # Sort by last_updated, most recent first
        sessions.sort(key=lambda x: x.get("last_updated", ""), reverse=True)

        return sessions

    def archive_to_gcs(self, session_id: str) -> bool:
        """Archive a session to GCS if BM is configured with a GCS save_dir.

        Args:
            session_id: The session identifier to archive

        Returns:
            bool: True if archival was successful, False otherwise
        """
        try:
            # Check if session exists
            if not self.session_exists(session_id):
                logger.warning(
                    "Cannot archive non-existent session", session_id=session_id
                )
                return False

            # Try to get BM instance to access save_dir
            try:
                # Check if save_dir is configured and points to GCS
                if not bm.session_info.save_dir:
                    logger.debug(
                        "No save_dir configured, skipping GCS archival for session",
                        session_id=session_id,
                    )
                    return False

                if not bm.session_info.save_dir.startswith(("gs://", "gcs://")):
                    logger.debug(
                        "save_dir is not GCS path, skipping archival for session",
                        session_id=session_id,
                    )
                    return False

            except Exception as e:
                logger.warning(
                    "Could not access BM instance for session archival", error=e
                )
                return False

            # Get session data
            session_data = self._get_or_create_session_data(session_id)

            # Add archival metadata
            session_data["archived_at"] = datetime.now(UTC).isoformat()
            session_data["archived_from"] = str(self._get_session_file(session_id))

            # Use BM's save method to archive to GCS
            archive_filename = f"session_{session_id}_archived.json"
            saved_path = bm.save(
                data=session_data,
                basename=f"sessions/{archive_filename}",
                extension="",  # Already included in basename
            )

            if saved_path:
                logger.info(
                    "Successfully archived session to GCS",
                    session_id=session_id,
                    saved_path=saved_path,
                )
                return True
            else:
                logger.error("Failed to archive session to GCS", session_id=session_id)
                return False

        except Exception as e:
            logger.error(
                "Error archiving session to GCS", session_id=session_id, error=e
            )
            return False

    def finalize_session(self, session_id: str, final_status: str) -> None:
        """Finalize a session and archive it to GCS.

        Called when a session reaches a terminal state (completed/failed).
        Updates the session with completion metadata and archives to GCS if configured.

        Args:
            session_id: The session identifier
            final_status: Final session status (completed or failed)

        Raises:
            IOError: If file cannot be written
        """
        try:
            # Append status update entry
            timestamp = datetime.now(UTC).isoformat()
            self._append_entry(
                session_id,
                {
                    "_type": "status_update",
                    "_timestamp": timestamp,
                    "status": final_status,
                },
            )

            # Append completion entry
            self._append_entry(
                session_id,
                {
                    "_type": "completion",
                    "_timestamp": timestamp,
                    "completed_at": timestamp,
                },
            )

            logger.info("Finalized session", session_id=session_id, status=final_status)

            # Always attempt archival for terminal states
            archive_success = self.archive_to_gcs(session_id)
            if archive_success:
                logger.info(
                    "Session archived to GCS after finalization", session_id=session_id
                )
            else:
                logger.debug(
                    "Session not archived (GCS not configured or archival failed)",
                    session_id=session_id,
                )

        except Exception as e:
            logger.error("Error finalizing session", session_id=session_id, error=e)
            raise
