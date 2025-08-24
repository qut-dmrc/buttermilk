import { writable } from 'svelte/store';
import type { Message } from '$lib/utils/messageUtils';
import { updateTokenUsage, resetTokenUsage } from './tokenUsageStore';

// Create a writable store for messages
export const messageStore = writable<Message[]>([]);

// Helper functions
export function addMessage(message: Message) {
	messageStore.update((messages) => {
		// Check for duplicate message_id to prevent duplicates
		if (message.message_id && messages.some(existing => existing.message_id === message.message_id)) {
			console.debug(`Skipping duplicate message with ID: ${message.message_id}`);
			return messages; // Return unchanged if duplicate found
		}
		return [...messages, message];
	});
	// Update token usage tracking (only for non-duplicates)
	updateTokenUsage(message);
}

export function clearMessages() {
	messageStore.set([]);
	// Reset token usage when clearing messages
	resetTokenUsage();
}
