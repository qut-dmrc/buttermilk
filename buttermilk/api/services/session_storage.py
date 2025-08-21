"""Session storage service for persisting chat flow messages to disk."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from buttermilk import logger
from buttermilk.api.services.message_service import ChatMessage

# Default sessions directory
SESSIONS_DIR = Path("data/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


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
                         Defaults to data/sessions/
        """
        self.sessions_dir = sessions_dir or SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Path to the session JSON file
        """
        return self.sessions_dir / f"{session_id}.json"

    def save_message(self, session_id: str, message: ChatMessage) -> None:
        """Save a message to the session file.
        
        Messages are appended to the existing session file if it exists,
        or a new file is created.
        
        Args:
            session_id: The session identifier
            message: The ChatMessage to persist
        """
        if not self.should_persist_message(message):
            logger.debug(f"Skipping persistence for message type: {message.type}")
            return

        session_file = self._get_session_file(session_id)
        
        try:
            # Load existing session data or create new
            if session_file.exists():
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Corrupted session file {session_file}, creating new")
                    session_data = self._create_new_session_data(session_id)
            else:
                session_data = self._create_new_session_data(session_id)
            
            # Add the new message
            message_dict = message.model_dump(mode="json")
            session_data["messages"].append(message_dict)
            session_data["last_updated"] = datetime.now().isoformat()
            
            # Write back to file
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logger.debug(f"Saved message {message.message_id} to session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save message to session {session_id}: {e}")

    def get_session_messages(self, session_id: str) -> List[ChatMessage]:
        """Retrieve all messages for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            List of ChatMessage objects, empty list if session doesn't exist
            or file is corrupted
        """
        session_file = self._get_session_file(session_id)
        
        if not session_file.exists():
            logger.debug(f"Session file not found: {session_id}")
            return []
        
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            messages = []
            for msg_data in session_data.get("messages", []):
                try:
                    message = ChatMessage(**msg_data)
                    messages.append(message)
                except Exception as e:
                    logger.warning(f"Failed to parse message in session {session_id}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages
            
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted session file {session_file}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to read session {session_id}: {e}")
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
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "messages": []
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
            logger.debug(f"Session file not found for deletion: {session_id}")
            return False
        
        try:
            session_file.unlink()
            logger.info(f"Deleted session file: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
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
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            return {
                "session_id": session_data.get("session_id"),
                "created_at": session_data.get("created_at"),
                "last_updated": session_data.get("last_updated"),
                "message_count": len(session_data.get("messages", []))
            }
        except Exception as e:
            logger.error(f"Failed to read session metadata for {session_id}: {e}")
            return None

    def list_sessions(self) -> List[dict]:
        """List all available sessions with metadata.
        
        Returns:
            List of session metadata dictionaries
        """
        sessions = []
        
        for session_file in self.sessions_dir.glob("*.json"):
            session_id = session_file.stem
            metadata = self.get_session_metadata(session_id)
            if metadata:
                sessions.append(metadata)
        
        # Sort by last_updated, most recent first
        sessions.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
        
        return sessions