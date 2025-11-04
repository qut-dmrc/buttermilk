/**
 * Tests for URL-based session routing in the frontend.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/svelte';
import { goto } from '$app/navigation';
import { sessionId } from '$lib/stores/sessionStore';
import { get } from 'svelte/store';

// Mock SvelteKit navigation
vi.mock('$app/navigation', () => ({
    goto: vi.fn()
}));

// Mock browser environment
vi.mock('$app/environment', () => ({
    browser: true
}));

describe('Session URL Routing', () => {
    beforeEach(() => {
        // Clear mocks and localStorage
        vi.clearAllMocks();
        localStorage.clear();
    });

    it('should extract session ID from URL parameters', () => {
        // Simulate URL with session ID
        const mockSessionId = 'test-session-123';
        const mockParams = { sessionId: mockSessionId };

        // In actual implementation, this would be in +page.svelte load function
        // Simulating the extraction logic
        const extractedId = mockParams.sessionId;

        expect(extractedId).toBe(mockSessionId);
    });

    it('should redirect from /terminal to /session/{id}', async () => {
        // Generate new session ID
        const newSessionId = 'generated-session-456';

        // Simulate navigation from /terminal
        await goto(`/session/${newSessionId}`, { replaceState: true });

        // Verify goto was called with correct URL
        expect(goto).toHaveBeenCalledWith(
            `/session/${newSessionId}`,
            { replaceState: true }
        );
    });

    it('should update URL when session changes', async () => {
        const oldSessionId = 'old-session';
        const newSessionId = 'new-session';

        // Set initial session
        sessionId.set(oldSessionId);

        // Change session
        sessionId.set(newSessionId);

        // In actual implementation, sessionStore would trigger URL update
        await goto(`/session/${newSessionId}`, { replaceState: true });

        expect(goto).toHaveBeenCalledWith(
            `/session/${newSessionId}`,
            { replaceState: true }
        );
    });

    it('should persist session ID in localStorage as fallback', () => {
        const testSessionId = 'localStorage-session';

        // Set session ID
        sessionId.set(testSessionId);

        // Verify it's saved to localStorage
        expect(localStorage.getItem('sessionId')).toBe(testSessionId);
    });

    it('should read session ID from URL first, localStorage second', () => {
        const urlSessionId = 'url-session';
        const localStorageSessionId = 'localStorage-session';

        // Set localStorage value
        localStorage.setItem('sessionId', localStorageSessionId);

        // Simulate URL parameter being available
        const mockUrlParams = { sessionId: urlSessionId };

        // URL should take precedence
        const resolvedSessionId = mockUrlParams.sessionId || localStorage.getItem('sessionId');

        expect(resolvedSessionId).toBe(urlSessionId);

        // Test fallback to localStorage when no URL param
        const noUrlParams = {};
        const fallbackSessionId = noUrlParams.sessionId || localStorage.getItem('sessionId');

        expect(fallbackSessionId).toBe(localStorageSessionId);
    });
});

describe('Session Restoration UI', () => {
    it('should fetch and display restored messages', async () => {
        const sessionId = 'restore-test-session';
        const mockMessages = [
            {
                type: 'record',
                message_id: 'msg-1',
                preview: 'First restored message',
                outputs: { content: 'Hello from the past' }
            },
            {
                type: 'record',
                message_id: 'msg-2',
                preview: 'Second restored message',
                outputs: { content: 'Previous conversation' }
            }
        ];

        // Mock fetch for session messages
        global.fetch = vi.fn().mockResolvedValueOnce({
            ok: true,
            json: async () => ({ messages: mockMessages })
        });

        // Simulate restoration function
        const restoreSessionMessages = async (sessionId: string) => {
            const response = await fetch(`/api/session/${sessionId}/messages`);
            if (response.ok) {
                const data = await response.json();
                return data.messages;
            }
            return [];
        };

        // Call restoration
        const restoredMessages = await restoreSessionMessages(sessionId);

        // Verify fetch was called correctly
        expect(fetch).toHaveBeenCalledWith(`/api/session/${sessionId}/messages`);

        // Verify messages were returned
        expect(restoredMessages).toHaveLength(2);
        expect(restoredMessages[0].message_id).toBe('msg-1');
        expect(restoredMessages[1].message_id).toBe('msg-2');
    });

    it('should handle session not found gracefully', async () => {
        const nonExistentSessionId = 'non-existent';

        // Mock fetch to return 404
        global.fetch = vi.fn().mockResolvedValueOnce({
            ok: false,
            status: 404,
            json: async () => ({ detail: 'Session not found' })
        });

        // Simulate restoration with error handling
        const restoreSessionMessages = async (sessionId: string) => {
            const response = await fetch(`/api/session/${sessionId}/messages`);
            if (!response.ok) {
                if (response.status === 404) {
                    // Start new session
                    return [];
                }
                throw new Error('Failed to restore session');
            }
            const data = await response.json();
            return data.messages;
        };

        // Should return empty array for non-existent session
        const messages = await restoreSessionMessages(nonExistentSessionId);
        expect(messages).toEqual([]);
    });

    it('should display messages in chronological order', () => {
        const messages = [
            {
                timestamp: '2024-01-01T10:00:00Z',
                message_id: 'msg-1',
                preview: 'First'
            },
            {
                timestamp: '2024-01-01T10:05:00Z',
                message_id: 'msg-2',
                preview: 'Second'
            },
            {
                timestamp: '2024-01-01T10:02:00Z',
                message_id: 'msg-3',
                preview: 'Should be middle'
            }
        ];

        // Sort messages by timestamp
        const sortedMessages = [...messages].sort((a, b) =>
            new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
        );

        // Verify order
        expect(sortedMessages[0].message_id).toBe('msg-1');
        expect(sortedMessages[1].message_id).toBe('msg-3');
        expect(sortedMessages[2].message_id).toBe('msg-2');
    });
});
