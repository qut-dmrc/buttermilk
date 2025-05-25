// Type definitions for Bootstrap 5.x
declare module 'bootstrap' {
  export class Tooltip {
    constructor(element: Element, options?: any);
    static getInstance(element: Element): Tooltip | null;
    dispose(): void;
    show(): void;
    hide(): void;
    toggle(): void;
    enable(): void;
    disable(): void;
  }
}

// Type definitions for Bootstrap bundle
declare module 'bootstrap/dist/js/bootstrap.bundle.min.js' {
  export * from 'bootstrap';
}
