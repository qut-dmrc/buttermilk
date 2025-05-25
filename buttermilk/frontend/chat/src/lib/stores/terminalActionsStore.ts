import { writable } from 'svelte/store';

// Type for the runFlow function signature (adjust if necessary)
type RunFlowFunction = (() => void) | null;

// Writable store to hold the runFlow function
export const runFlowAction = writable<RunFlowFunction>(null);
