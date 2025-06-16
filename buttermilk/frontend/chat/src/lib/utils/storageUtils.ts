/**
 * Utility functions for managing localStorage and storage cleanup
 */

export function clearAllMessageHistory(): void {
    if (typeof window === 'undefined') return;
    
    try {
        const keys = Object.keys(localStorage);
        const messageKeys = keys.filter(key => 
            key.startsWith('messageHistory_') || key === 'messageHistory'
        );
        
        messageKeys.forEach(key => {
            localStorage.removeItem(key);
        });
        
        console.log(`Cleared ${messageKeys.length} message history entries from localStorage`);
        
        // Also try to clean up any other Buttermilk-related storage
        const buttermilkKeys = keys.filter(key => 
            key.includes('buttermilk') || 
            key.includes('session') ||
            key.includes('flow')
        );
        
        buttermilkKeys.forEach(key => {
            if (!key.includes('sessionId')) { // Keep the current session ID
                localStorage.removeItem(key);
            }
        });
        
        if (buttermilkKeys.length > 0) {
            console.log(`Cleared ${buttermilkKeys.length} additional storage entries`);
        }
        
    } catch (e) {
        console.error('Error during storage cleanup:', e);
    }
}

export function getStorageUsage(): { used: number; total: number; percentage: number } {
    if (typeof window === 'undefined') {
        return { used: 0, total: 0, percentage: 0 };
    }
    
    try {
        let used = 0;
        for (const key in localStorage) {
            if (localStorage.hasOwnProperty(key)) {
                used += localStorage[key].length + key.length;
            }
        }
        
        // Estimate total available (usually 5-10MB, but varies by browser)
        const total = 5 * 1024 * 1024; // 5MB estimate
        const percentage = (used / total) * 100;
        
        return { used, total, percentage };
    } catch (e) {
        console.error('Error calculating storage usage:', e);
        return { used: 0, total: 0, percentage: 0 };
    }
}

export function getMessageHistoryStorageInfo(): { 
    sessions: number; 
    totalSize: number; 
    largestSession: { key: string; size: number } | null 
} {
    if (typeof window === 'undefined') {
        return { sessions: 0, totalSize: 0, largestSession: null };
    }
    
    try {
        const keys = Object.keys(localStorage);
        const messageKeys = keys.filter(key => key.startsWith('messageHistory_'));
        
        let totalSize = 0;
        let largestSession: { key: string; size: number } | null = null;
        
        messageKeys.forEach(key => {
            const value = localStorage.getItem(key);
            if (value) {
                const size = value.length + key.length;
                totalSize += size;
                
                if (!largestSession || size > largestSession.size) {
                    largestSession = { key, size };
                }
            }
        });
        
        return {
            sessions: messageKeys.length,
            totalSize,
            largestSession
        };
    } catch (e) {
        console.error('Error analyzing message history storage:', e);
        return { sessions: 0, totalSize: 0, largestSession: null };
    }
}

// Format bytes for human reading
export function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}