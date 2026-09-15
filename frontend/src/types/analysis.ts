export type AnalysisStatus = 'matched' | 'no_template_match' | 'no_card_detected' | 'extraction_failed'

export interface Analysis {
  status: AnalysisStatus
  card_detected: boolean
  template_id: string | null
  template_name: string | null
  detection_confidence: number | null
  match_score: number | null
  match_threshold: number
  is_supported: boolean | null
  corners: { x: number; y: number }[] | null
  image_width: number
  image_height: number
  extracted_card: string | null
  failure_stage: 'crop' | 'refinement' | null
  processing_time_ms: number
  inference_time_ms: number
}
