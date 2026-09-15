import type { Analysis } from '../types/analysis'

export async function analyzeImage(file: File, signal: AbortSignal): Promise<Analysis> {
  const body = new FormData()
  body.append('file', file)
  let response: Response
  try {
    response = await fetch('/api/v1/analyze', { method: 'POST', body, signal })
  } catch (error) {
    if (signal.aborted) throw error
    throw new Error('Cannot reach the analysis service. Check that the backend is running and try again.')
  }
  if (!response.ok) {
    const failure = await response.json().catch(() => null)
    throw new Error(failure?.error?.message ?? (response.status === 413
      ? 'The image is too large. Choose a file under 10 MB.'
      : `The service could not complete this request (${response.status}). Please try again.`))
  }
  return response.json() as Promise<Analysis>
}
