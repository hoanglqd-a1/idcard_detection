import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  outputDir: '../artifacts/browser',
  timeout: 120_000,
  use: {
    baseURL: process.env.CARDSCOPE_URL ?? 'http://127.0.0.1:8080',
    ...devices['Desktop Edge'],
    channel: 'msedge',
    headless: true,
  },
  workers: 1,
})
