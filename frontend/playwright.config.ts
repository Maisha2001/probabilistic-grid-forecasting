import {defineConfig} from '@playwright/test';
const external = process.env.GRID_DEMO_URL;
export default defineConfig({
  testDir: './tests', workers: 1, timeout: 30000,
  use: {baseURL: external || 'http://127.0.0.1:8013', browserName: 'chromium', headless: true},
  webServer: external ? undefined : {command: `"${process.env.GRID_PYTHON || 'python'}" ../scripts/serve_fixture.py`, url: 'http://127.0.0.1:8013/api/health', timeout: 120000},
  reporter: [['list'], ['json', {outputFile:'test-results/results.json'}]]
});
