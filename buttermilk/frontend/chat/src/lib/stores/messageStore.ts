import { writable } from 'svelte/store';
import type { Message } from '$lib/utils/messageUtils';

// Create a writable store for messages
export const messageStore = writable<Message[]>([]);

// Helper functions
export function addMessage(message: Message) {
  messageStore.update(messages => [...messages, message]);
}

export function clearMessages() {
  messageStore.set([]);
}
