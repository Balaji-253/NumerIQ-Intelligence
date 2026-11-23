export interface DigitPrediction {
  position: number
  digit: string
  confidence: number
  is_uncertain: boolean
  bbox: number[]
  area: number
  component_count: number
  top3: unknown[]
}

export interface SequencePrediction {
  status: string
  sequence: string
  digit_count: number
  digits: DigitPrediction[]
  warnings: string[]
}

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
).replace(/\/$/, '')

export async function recognizeSequence(
  blob: Blob,
  signal?: AbortSignal,
): Promise<SequencePrediction> {
  const formData = new FormData()

  formData.append('file', blob, 'numeriq-canvas.png')

  const response = await fetch(
    `${API_BASE_URL}/predict/sequence`,
    {
      method: 'POST',
      body: formData,
      signal,
    },
  )

  const payload = await response.json().catch(() => null)

  if (!response.ok) {
    throw new Error(
      payload?.detail || `Recognition failed (${response.status})`,
    )
  }

  return payload as SequencePrediction
}
