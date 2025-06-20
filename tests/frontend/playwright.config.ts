import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright Configuration for Buttermilk Terminal Interface Testing
 * 
 * Comprehensive E2E testing configuration supporting:
 * - Multiple browser engines (Chromium, Firefox, WebKit)
 * - Mobile device simulation
 * - Performance testing and visual regression
 * - Parallel execution and retries
 * - Custom test environments and reporting
 */

export default defineConfig({
  // Test directory
  testDir: '.',
  
  // Run tests in files matching this pattern
  testMatch: '**/*.spec.ts',
  
  // Timeout for each test
  timeout: 60000,
  
  // Global setup and teardown
  expect: {
    // Timeout for expect() assertions
    timeout: 10000,
  },
  
  // Test execution configuration
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 1,
  workers: process.env.CI ? 1 : undefined,
  
  // Reporting configuration
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'test-results.json' }],
    ['junit', { outputFile: 'junit-results.xml' }],
    ['list']
  ],
  
  // Global test settings
  use: {
    // Base URL for the frontend application
    baseURL: process.env.FRONTEND_URL || 'http://localhost:5173',
    
    // Browser context options
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    
    // Network settings
    extraHTTPHeaders: {
      'Accept': 'application/json',
    },
    
    // Ignore HTTPS errors in development
    ignoreHTTPSErrors: true,
  },

  // Project configurations for different browsers and devices
  projects: [
    // Desktop Browsers
    {
      name: 'chromium',
      use: { 
        ...devices['Desktop Chrome'],
        // Additional Chrome-specific settings
        launchOptions: {
          args: ['--disable-web-security', '--disable-features=VizDisplayCompositor'],
        },
      },
    },
    
    {
      name: 'firefox',
      use: { 
        ...devices['Desktop Firefox'],
        // Firefox-specific settings
        launchOptions: {
          firefoxUserPrefs: {
            'dom.webnotifications.enabled': false,
          },
        },
      },
    },
    
    {
      name: 'webkit',
      use: { 
        ...devices['Desktop Safari'],
        // WebKit-specific settings
      },
    },

    // Mobile Browsers
    {
      name: 'Mobile Chrome',
      use: { 
        ...devices['Pixel 5'],
      },
    },
    
    {
      name: 'Mobile Safari',
      use: { 
        ...devices['iPhone 12'],
      },
    },

    // Performance Testing
    {
      name: 'performance',
      use: {
        ...devices['Desktop Chrome'],
        // Throttle network for performance testing
        launchOptions: {
          args: ['--disable-web-security'],
        },
      },
      testMatch: '**/*performance*.spec.ts',
    },

    // Accessibility Testing
    {
      name: 'accessibility',
      use: {
        ...devices['Desktop Chrome'],
        // Enable accessibility features
        launchOptions: {
          args: ['--force-prefers-reduced-motion', '--enable-accessibility-logging'],
        },
      },
      testMatch: '**/*accessibility*.spec.ts',
    },
  ],

  // Web server configuration for local development
  webServer: [
    {
      // Frontend SvelteKit dev server
      command: 'npm run dev',
      url: 'http://localhost:5173',
      cwd: '../../buttermilk/frontend/chat',
      // reuseExistingServer: !process.env.CI,
      reuseExistingServer: true, 
      timeout: 120000,
      env: {
        NODE_ENV: 'test',
      },
    },
    {
      // Backend FastAPI server
      command: 'python -m uvicorn buttermilk.api.flow:create_app --reload --host 0.0.0.0 --port 8000',
      url: 'http://localhost:8000',
      cwd: '../..',
      reuseExistingServer: !process.env.CI,
      // reuseExistingServer: true,  // Add this line
      timeout: 120000,
      env: {
        PYTHONPATH: '../../',
        ENVIRONMENT: 'test',
      },
    },
  ],

  // Test output directories
  outputDir: 'test-results/',
  
  // Global test setup
  globalSetup: require.resolve('./global-setup.ts'),
  globalTeardown: require.resolve('./global-teardown.ts'),
});