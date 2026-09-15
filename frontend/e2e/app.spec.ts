import { expect, test } from '@playwright/test'
import path from 'node:path'

test('real upload through the frontend proxy, with desktop and mobile layouts', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/')
  await expect(page.getByRole('heading', { name: /From an image/ })).toBeVisible()
  await page.screenshot({ path: '../artifacts/browser/desktop-empty.png', fullPage: true })
  await page.getByLabel('Upload document image').setInputFiles(path.resolve('../training/test_images/image6.png'))
  await expect(page.getByAltText('Original uploaded document')).toBeVisible()
  const response = page.waitForResponse(response => response.url().endsWith('/api/v1/analyze') && response.request().method() === 'POST')
  await page.getByRole('button', { name: /Analyze image/ }).click()
  const analyzed = await response
  expect(analyzed.status()).toBe(200)
  const result = await analyzed.json()
  expect(result.card_detected).toBe(true)
  expect(result.extracted_card).toMatch(/^data:image\/png;base64,/)
  await expect(page.getByAltText('Extracted and rectified card')).toBeVisible()
  await expect(page.getByLabel('Detected card region')).toBeVisible()
  await page.screenshot({ path: '../artifacts/browser/desktop-result.png', fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.screenshot({ path: '../artifacts/browser/mobile-result.png', fullPage: true })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  expect(errors).toEqual([])
})
