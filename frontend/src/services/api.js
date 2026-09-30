export async function predictImage(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch('/api/predict', {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`
    try {
      const data = await response.json()
      if (data.detail) detail = data.detail
    } catch {
      // response had no JSON body
    }
    throw new Error(detail)
  }

  return response.json()
}

export async function getModelInfo() {
  const response = await fetch('/api/model-info')
  if (!response.ok) throw new Error('Backend unreachable')
  return response.json()
}