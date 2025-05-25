import { browser } from '$app/environment';
import { writable } from 'svelte/store';

// Create a writable store with localStorage persistence
function createSessionStore() {
    // Default empty session ID
    const defaultValue = '';
    
    // Read initial value from localStorage if available
    const initialValue = browser ? 
        window.localStorage.getItem('sessionId') || defaultValue : 
        defaultValue;
    
    // Create writable store with the initial value
    const sessionStore = writable<string>(initialValue);
    
    // Subscribe to changes and update localStorage
    if (browser) {
        sessionStore.subscribe(value => {
            if (value) {
                window.localStorage.setItem('sessionId', value);
            } else {
                window.localStorage.removeItem('sessionId');
            }
        });
    }
    
    return {
        ...sessionStore,
        // Method to clear the session
        clear: () => {
            sessionStore.set('');
            if (browser) {
                window.localStorage.removeItem('sessionId');
            }
        }
    };
}

export const sessionId = createSessionStore();
