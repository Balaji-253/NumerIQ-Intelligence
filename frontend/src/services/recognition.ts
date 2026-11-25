export interface DigitPrediction {
  position: number
  digit: string
  confidence: number
  is_uncertain: boolean
  bbox: number[]
  area: number
  component_count?: number
  top3: unknown[]
}

export interface SequencePrediction {
  status: string
  sequence: string
  digit_count: number
  digits: DigitPrediction[]
  warnings: string[]
}

export interface SingleDigitPrediction {
  digit: number
  confidence: number
  is_uncertain: boolean
  top3: unknown[]
  top5: unknown[]
  model: string
  inference_device: string
  latency_ms: number
}

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
).replace(/\/$/, '')

async function uploadImage<T>(
  endpoint: string,
  blob: Blob,
  signal?: AbortSignal,
): Promise<T> {
  const formData = new FormData()
  formData.append('file', blob, 'numeriq-upload.png')

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'POST',
    body: formData,
    signal,
  })

  const payload = await response.json().catch(() => null)

  if (!response.ok) {
    throw new Error(
      typeof payload?.detail === 'string'
        ? payload.detail
        : `Recognition failed (${response.status})`,
    )
  }

  return payload as T
}

export function recognizeSequence(
  blob: Blob,
  signal?: AbortSignal,
): Promise<SequencePrediction> {
  return uploadImage('/predict/sequence', blob, signal)
}

export function recognizeSingleDigit(
  blob: Blob,
  signal?: AbortSignal,
): Promise<SingleDigitPrediction> {
  return uploadImage('/predict', blob, signal)
}

export async function checkApiHealth(): Promise<boolean> {
  const response = await fetch(`${API_BASE_URL}/ready`)
  if (!response.ok) return false

  const result = await response.json()
  return result.status === 'ready' && result.model_loaded === true
}
