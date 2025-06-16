import { browser } from '$app/environment';
import { writable } from 'svelte/store';
import type { Message } from '$lib/utils/messageUtils';

const MAX_STORED_MESSAGES = 50; // Reduced limit for performance
const MAX_CONTENT_LENGTH = 1000; // Limit content size for storage

// Lightweight message for storage (strip large fields)
interface StoredMessage {
    type: string;
    message_id: string;
    preview: string;
    timestamp: string;
    agent_name?: string;
    agent_role?: string;
    model?: string;
}

function createMessageHistoryStore() {
    // Try to load stored messages
    let initialMessages: Message[] = [];
    if (browser) {
        try {
            const stored = localStorage.getItem('messageHistory');
            if (stored) {
                const storedMessages: StoredMessage[] = JSON.parse(stored);
                // Convert stored messages back to full messages (with limited data)
                initialMessages = storedMessages.map(sm => ({
                    type: sm.type as any,
                    message_id: sm.message_id,
                    preview: sm.preview,
                    timestamp: sm.timestamp,
                    agent_info: sm.agent_name ? {
                        agent_id: '',
                        session_id: '',
                        role: sm.agent_role || '',
                        agent_name: sm.agent_name,
                        description: '',
                        parameters: { model: sm.model }
                    } : undefined
                }));
            }
        } catch (e) {
            console.error('Error loading stored messages:', e);
            // Clear corrupted storage
            if (browser) {
                localStorage.removeItem('messageHistory');
            }
        }
    }
    
    const store = writable<Message[]>(initialMessages);
    
    // Convert message to lightweight storage format
    function createStoredMessage(message: Message): StoredMessage {
        return {
            type: message.type,
            message_id: message.message_id,
            preview: (message.preview || '').substring(0, MAX_CONTENT_LENGTH),
            timestamp: message.timestamp,
            agent_name: message.agent_info?.agent_name,
            agent_role: message.agent_info?.role,
            model: message.agent_info?.parameters?.model
        };
    }
    
    // Clean up old session storage to prevent quota issues
    function cleanupOldSessions() {
        if (!browser) return;
        
        try {
            const keys = Object.keys(localStorage);
            const messageHistoryKeys = keys.filter(key => key.startsWith('messageHistory_'));
            
            // If we have too many sessions, remove the oldest ones
            if (messageHistoryKeys.length > 10) {
                console.log(`Found ${messageHistoryKeys.length} stored sessions, cleaning up old ones`);
                
                // Sort by key (which includes timestamp info) and remove oldest
                messageHistoryKeys.sort();
                const keysToRemove = messageHistoryKeys.slice(0, messageHistoryKeys.length - 5);
                
                keysToRemove.forEach(key => {
                    localStorage.removeItem(key);
                    console.log(`Removed old session storage: ${key}`);
                });
            }
        } catch (e) {
            console.error('Error cleaning up old sessions:', e);
        }
    }
    
    return {
        ...store,
        addMessage: (message: Message) => {
            store.update(messages => {
                const newMessages = [...messages, message];
                // Limit size to prevent localStorage issues
                if (newMessages.length > MAX_STORED_MESSAGES) {
                    newMessages.shift(); // Remove oldest message
                }
                
                if (browser) {
                    try {
                        // Store only essential data in lightweight format
                        const lightweightMessages = newMessages.map(createStoredMessage);
                        localStorage.setItem('messageHistory', JSON.stringify(lightweightMessages));
                    } catch (e) {
                        console.error('Error storing messages:', e);
                        // If storage fails, try to clear old data and retry
                        cleanupOldSessions();
                        try {
                            const lightweightMessages = newMessages.slice(-20).map(createStoredMessage); // Keep only last 20
                            localStorage.setItem('messageHistory', JSON.stringify(lightweightMessages));
                        } catch (e2) {
                            console.error('Failed to store messages even after cleanup:', e2);
                        }
                    }
                }
                
                return newMessages;
            });
        },
        clearHistory: () => {
            store.set([]);
            if (browser) {
                localStorage.removeItem('messageHistory');
            }
        },
        // Associate with a specific session
        setSession: (sessionId: string) => {
            if (browser) {
                try {
                    cleanupOldSessions(); // Clean up when switching sessions
                    
                    const key = `messageHistory_${sessionId}`;
                    const stored = localStorage.getItem(key);
                    if (stored) {
                        const storedMessages: StoredMessage[] = JSON.parse(stored);
                        // Convert back to full messages
                        const messages = storedMessages.map(sm => ({
                            type: sm.type as any,
                            message_id: sm.message_id,
                            preview: sm.preview,
                            timestamp: sm.timestamp,
                            agent_info: sm.agent_name ? {
                                agent_id: '',
                                session_id: '',
                                role: sm.agent_role || '',
                                agent_name: sm.agent_name,
                                description: '',
                                parameters: { model: sm.model }
                            } : undefined
                        }));
                        store.set(messages);
                    } else {
                        store.set([]);
                    }
                } catch (e) {
                    console.error('Error loading session messages:', e);
                    store.set([]);
                }
            }
        },
        // Save messages for a specific session (with lightweight format)
        saveForSession: (sessionId: string, messages: Message[]) => {
            if (browser && sessionId) {
                try {
                    cleanupOldSessions(); // Clean up before saving
                    
                    const key = `messageHistory_${sessionId}`;
                    // Convert to lightweight format and limit number
                    const limitedMessages = messages.slice(-MAX_STORED_MESSAGES);
                    const lightweightMessages = limitedMessages.map(createStoredMessage);
                    
                    localStorage.setItem(key, JSON.stringify(lightweightMessages));
                } catch (e) {
                    console.error('Error saving session messages:', e);
                    
                    // If storage fails due to quota, try emergency cleanup
                    try {
                        // Clear ALL old message history
                        const keys = Object.keys(localStorage);
                        keys.filter(key => key.startsWith('messageHistory_')).forEach(key => {
                            localStorage.removeItem(key);
                        });
                        
                        // Try to save just the last few messages
                        const emergencyMessages = messages.slice(-10).map(createStoredMessage);
                        localStorage.setItem(key, JSON.stringify(emergencyMessages));
                        console.log('Emergency cleanup completed, saved last 10 messages');
                    } catch (e2) {
                        console.error('Emergency cleanup also failed:', e2);
                    }
                }
            }
        }
    };
}

export const messageHistory = createMessageHistoryStore();
