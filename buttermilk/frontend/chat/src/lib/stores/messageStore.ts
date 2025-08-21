import { writable } from 'svelte/store';
import type { Message } from '$lib/utils/messageUtils';
import { updateTokenUsage, resetTokenUsage } from './tokenUsageStore';

// Create a writable store for messages
export const messageStore = writable<Message[]>([]);

// Helper functions
export function addMessage(message: Message) {
	messageStore.update((messages) => [...messages, message]);
	// Update token usage tracking
	updateTokenUsage(message);
}

export function clearMessages() {
	messageStore.set([]);
	// Reset token usage when clearing messages
	resetTokenUsage();
}
