import { useEffect, useRef, useState } from 'react'
import { analyzeImage } from './api/client'
import { Results } from './components/Results'
import type { Analysis } from './types/analysis'

const MAX_BYTES = 10 * 1024 * 1024

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<Analysis | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const request = useRef<AbortController | null>(null)
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!file) { setPreview(null); return }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])
  useEffect(() => () => request.current?.abort(), [])

  function selectFile(files: FileList | null) {
    request.current?.abort()
    request.current = null
    setBusy(false); setResult(null); setError(null); setFile(null)
    if (!files?.length) return
    if (files.length !== 1) { setError('Choose one image at a time.'); return }
    const next = files[0]
    if (next.size === 0) { setError('This file is empty. Choose another image.'); return }
    if (next.size > MAX_BYTES) { setError('Choose an image under 10 MB.'); return }
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(next.type)) {
      setError('Choose a JPEG, PNG, or WebP image.'); return
    }
    setFile(next)
  }

  async function analyze() {
    if (!file || busy) return
    const controller = new AbortController()
    request.current = controller
    setBusy(true); setError(null); setResult(null)
    try {
      const response = await analyzeImage(file, controller.signal)
      if (request.current === controller) setResult(response)
    } catch (failure) {
      if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : 'Analysis failed. Please try again.')
    } finally {
      if (request.current === controller) { setBusy(false); request.current = null }
    }
  }

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="./" aria-label="CardScope home"><span className="brand-icon">▱</span>CardScope<span className="brand-tag">VISION LAB</span></a>
      <span className="demo-label"><span />Portfolio demo</span>
    </header>
    <main>
      <section className="intro">
        <span className="eyebrow">DOCUMENT INTELLIGENCE / 01</span>
        <h1>From an image<br />to a <span>clearer picture.</span></h1>
        <p>Detect a document card, straighten its perspective, and compare it with reference templates. Explore the vision pipeline, one image at a time.</p>
        <div className="intro-tags"><span>Card detection</span><span>Perspective correction</span><span>Template matching</span></div>
      </section>
      <section className="workspace" aria-label="Document analysis workspace">
        <aside className="upload-panel">
          <div className="section-label"><span>01</span><h2>Your image</h2></div>
          <p className="muted">Start with a photo containing one document card.</p>
          <div className={`dropzone ${dragging ? 'dragging' : ''}`}
            onDragOver={event => { event.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={event => { event.preventDefault(); setDragging(false); selectFile(event.dataTransfer.files) }}>
            <svg width="42" height="42" viewBox="0 0 40 40" fill="none" aria-hidden="true"><rect x="5" y="9" width="30" height="24" rx="5" stroke="currentColor" strokeWidth="1.5"/><path d="M20 26V15m-5 5 5-5 5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
            <strong>Drop your image here</strong>
            <span className="muted">or choose a file to get started</span>
            <button className="secondary" onClick={() => input.current?.click()}>Browse files</button>
            <input ref={input} className="sr-only" type="file" aria-label="Upload document image"
              accept="image/jpeg,image/png,image/webp" onChange={event => { selectFile(event.target.files); event.target.value = '' }} />
            <small>JPEG, PNG, WebP · up to 10 MB</small>
          </div>
          {file && <div className="file-info"><span title={file.name}>{file.name}</span><small>{(file.size / 1024 / 1024).toFixed(2)} MB</small></div>}
          <button className="primary" disabled={!file || busy} onClick={analyze}>{busy ? <><span className="spinner" />Analyzing image…</> : <>Analyze image <span aria-hidden="true">↗</span></>}</button>
          <div aria-live="polite">{busy && <p className="footnote">Running detection, extraction, and matching. The first analysis may take longer.</p>}</div>
          {error && <p className="error" role="alert">{error}</p>}
          <div className="upload-note"><strong>A good image goes a long way.</strong><p>Keep all four corners visible. Avoid glare and use an upright, well-lit photo.</p></div>
        </aside>
        <div className="visual-panel" aria-busy={busy}>
          <div className="section-label"><span>02</span><h2>See the pipeline</h2></div>
          <div className="image-grid">
            <figure><figcaption>Original image <span>{result?.corners ? 'Detected region' : 'Input'}</span></figcaption>
              <div className="image-stage">
                {preview ? <div className="image-overlay">
                  <img src={preview} alt="Original uploaded document" onError={() => { setError('This image cannot be previewed. Choose another file.'); setFile(null); setResult(null); request.current?.abort() }} />
                  {result?.corners && <svg viewBox={`0 0 ${result.image_width} ${result.image_height}`} aria-label="Detected card region">
                    <polygon points={result.corners.map(point => `${point.x},${point.y}`).join(' ')} fill="#28b58722" stroke="#0d926b" strokeWidth="3" vectorEffect="non-scaling-stroke" />
                  </svg>}
                </div> : <div className="placeholder"><span className="outline-card" /><p>Your original image<br /><small>will appear here</small></p></div>}
              </div>
            </figure>
            <figure><figcaption>Extracted card <span>Rectified</span></figcaption>
              <div className="image-stage">
                {result?.extracted_card ? <img src={result.extracted_card} alt="Extracted and rectified card" /> : <div className="placeholder"><span className={`outline-card flat ${busy ? 'scanning' : ''}`} /><p>{busy ? 'Finding the card…' : result ? 'No extracted card' : 'A closer look'}<br /><small>{result ? 'See the analysis details below' : 'Detected, cropped, and straightened'}</small></p></div>}
              </div>
            </figure>
          </div>
          <div aria-live="polite">{result ? <Results result={result} /> : <div className="pipeline-note"><span>THE PROCESS</span><p>Detect the region <b>→</b> Rectify the card <b>→</b> Match a reference</p><small>Results and similarity scores appear after analysis.</small></div>}</div>
        </div>
      </section>
      <footer><span>CardScope <span className="footer-divider">/</span> Computer vision, made visible.</span><p>Reference matching demo · No OCR or identity verification · Uploads are not retained</p></footer>
    </main>
  </div>
}
