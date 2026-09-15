import type { Analysis } from '../types/analysis'

const outcomes = {
  matched: ['Reference matched', 'The extracted card matched a reference template.'],
  no_template_match: ['No reference template matched', 'A card was extracted, but its similarity did not exceed the matching threshold. Try a clearer, upright image.'],
  no_card_detected: ['No card detected', 'Try a well-lit photo with the whole card visible and a clear background.'],
  extraction_failed: ['Card detected · extraction incomplete', 'A region was detected, but the card could not be rectified. Try a flatter angle with all four edges visible.'],
} as const

export function Results({ result }: { result: Analysis }) {
  const [title, description] = outcomes[result.status]
  const stages = [
    { name: 'Detect', done: result.card_detected },
    { name: 'Extract', done: result.extracted_card !== null },
    { name: 'Match', done: result.is_supported === true },
  ]
  return <section className="result-panel" aria-label="Analysis result">
    <div className="result-heading">
      <div><span className="eyebrow">ANALYSIS RESULT</span><h2>{title}</h2></div>
      <span className={`badge ${result.is_supported ? 'success' : ''}`}>
        {result.is_supported === true ? 'Supported reference' : result.is_supported === false ? 'Unmatched' : 'Incomplete'}
      </span>
    </div>
    <p className="muted">{description}</p>
    <ol className="stages">{stages.map((stage, i) => <li key={stage.name} className={stage.done ? 'done' : ''}>
      <span>{stage.done ? '✓' : `0${i + 1}`}</span>{stage.name}
    </li>)}</ol>
    <dl className="metrics">
      <div><dt>Matched template</dt><dd>{result.template_name ?? '—'}</dd></div>
      <div><dt>Detection confidence</dt><dd>{result.detection_confidence == null ? '—' : `${(result.detection_confidence * 100).toFixed(1)}%`}</dd></div>
      <div><dt>Template similarity</dt><dd>{result.match_score == null ? '—' : result.match_score.toFixed(3)}</dd></div>
      <div><dt>Server processing</dt><dd>{result.processing_time_ms.toFixed(0)} <small>ms</small></dd></div>
    </dl>
    <p className="footnote">A reference match requires similarity &gt; {result.match_threshold}. Similarity is a correlation score, not a probability of correct classification.</p>
  </section>
}
