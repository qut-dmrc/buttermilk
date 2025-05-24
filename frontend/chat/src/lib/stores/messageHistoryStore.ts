import { browser } from '$app/environment';
import { writable } from 'svelte/store';
import type { Message } from '$lib/utils/messageUtils';

const MAX_STORED_MESSAGES = 100; // Limit for performance

function createMessageHistoryStore() {
    // Try to load stored messages
    let initialMessages: Message[] = [];
    if (browser) {
        try {
            const stored = localStorage.getItem('messageHistory');
            if (stored) {
                initialMessages = JSON.parse(stored);
            }
        } catch (e) {
            console.error('Error loading stored messages:', e);
        }
    }
    
    const store = writable<Message[]>(initialMessages);
    
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
                        localStorage.setItem('messageHistory', JSON.stringify(newMessages));
                    } catch (e) {
                        console.error('Error storing messages:', e);
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
                    const key = `messageHistory_${sessionId}`;
                    const stored = localStorage.getItem(key);
                    if (stored) {
                        store.set(JSON.parse(stored));
                    } else {
                        store.set([]);
                    }
                } catch (e) {
                    console.error('Error loading session messages:', e);
                    store.set([]);
                }
            }
        },
        // Save messages for a specific session
        saveForSession: (sessionId: string, messages: Message[]) => {
            if (browser && sessionId) {
                try {
                    const key = `messageHistory_${sessionId}`;
                    localStorage.setItem(key, JSON.stringify(messages));
                } catch (e) {
                    console.error('Error saving session messages:', e);
                }
            }
        }
    };
}

export const messageHistory = createMessageHistoryStore();
