import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'

const result = {
  status: 'matched', card_detected: true, template_id: 'Template 3.jpg', template_name: 'Template 3',
  detection_confidence: 0.94, match_score: 0.87, match_threshold: 0.8, is_supported: true,
  corners: [{ x: 0, y: 0 }, { x: 90, y: 0 }, { x: 90, y: 45 }, { x: 0, y: 45 }],
  image_width: 100, image_height: 50, extracted_card: 'data:image/png;base64,abc',
  failure_stage: null, processing_time_ms: 25, inference_time_ms: 20,
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn())
  URL.createObjectURL = vi.fn(() => 'blob:preview')
  URL.revokeObjectURL = vi.fn()
})
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

async function upload() {
  const user = userEvent.setup()
  await user.upload(screen.getByLabelText('Upload document image'), new File(['image'], 'card.png', { type: 'image/png' }))
  return user
}

describe('document analysis', () => {
  it('uploads an image and displays real API fields', async () => {
    vi.mocked(fetch).mockResolvedValue({ ok: true, json: async () => result } as Response)
    render(<App />)
    expect((screen.getByRole('button', { name: /Analyze image/ }) as HTMLButtonElement).disabled).toBe(true)
    const user = await upload()
    await user.click(screen.getByRole('button', { name: /Analyze image/ }))
    await screen.findByText('Template 3')
    expect(screen.getByText('0.870')).toBeTruthy()
    expect(screen.getByText('94.0%')).toBeTruthy()
    expect(screen.getByLabelText('Detected card region')).toBeTruthy()
    expect(screen.getByAltText('Extracted and rectified card')).toBeTruthy()
    const options = vi.mocked(fetch).mock.calls[0][1]!
    expect((options.body as FormData).get('file')).toBeInstanceOf(File)
  })

  it.each([
    ['no_template_match', 'No reference template matched'],
    ['no_card_detected', 'No card detected'],
    ['extraction_failed', 'Card detected · extraction incomplete'],
  ])('renders %s', async (status, message) => {
    vi.mocked(fetch).mockResolvedValue({ ok: true, json: async () => ({ ...result, status,
      template_id: null, template_name: null, is_supported: status === 'no_template_match' ? false : null,
      extracted_card: status === 'no_template_match' ? result.extracted_card : null,
    }) } as Response)
    render(<App />)
    const user = await upload()
    await user.click(screen.getByRole('button', { name: /Analyze image/ }))
    expect(await screen.findByText(message)).toBeTruthy()
  })

  it('clears stale results when choosing another image', async () => {
    vi.mocked(fetch).mockResolvedValue({ ok: true, json: async () => result } as Response)
    render(<App />)
    const user = await upload()
    await user.click(screen.getByRole('button', { name: /Analyze image/ }))
    await screen.findByText('Template 3')
    await user.upload(screen.getByLabelText('Upload document image'), new File(['next'], 'next.png', { type: 'image/png' }))
    expect(screen.queryByText('Template 3')).toBeNull()
    expect(URL.revokeObjectURL).toHaveBeenCalled()
  })

  it('shows loading and a useful server error', async () => {
    let finish!: (value: Response) => void
    vi.mocked(fetch).mockReturnValue(new Promise(resolve => { finish = resolve }))
    render(<App />)
    const user = await upload()
    await user.click(screen.getByRole('button', { name: /Analyze image/ }))
    expect((screen.getByRole('button', { name: /Analyzing image/ }) as HTMLButtonElement).disabled).toBe(true)
    finish({ ok: false, status: 500, json: async () => ({ error: { message: 'Please retry.' } }) } as Response)
    expect((await screen.findByRole('alert')).textContent).toContain('Please retry.')
    await waitFor(() => expect((screen.getByRole('button', { name: /Analyze image/ }) as HTMLButtonElement).disabled).toBe(false))
  })

  it('rejects multiple dropped files before calling the API', () => {
    render(<App />)
    fireEvent.drop(screen.getByText('Drop your image here').parentElement!, { dataTransfer: { files: [new File(['x'], 'a.png'), new File(['x'], 'b.png')] } })
    expect(screen.getByRole('alert').textContent).toContain('one image')
    expect(fetch).not.toHaveBeenCalled()
  })
})
