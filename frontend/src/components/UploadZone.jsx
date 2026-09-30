import { useRef, useState } from 'react'

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']
const MAX_SIZE = 10 * 1024 * 1024   // keep in sync with backend

export default function UploadZone({ onFile, onError }) {
  const inputRef = useRef(null)
  const [dragActive, setDragActive] = useState(false)

  const validate = (file) => {
    if (!ALLOWED_TYPES.includes(file.type))
      return 'Unsupported format. Use JPEG, PNG, WEBP or BMP.'
    if (file.size > MAX_SIZE)
      return 'File too large (max 10 MB).'
    return null
  }

  const handleFile = (file) => {
    if (!file) return
    const error = validate(file)
    error ? onError(error) : onFile(file)
  }

  return (
    <div
      className={`upload-zone ${dragActive ? 'active' : ''}`}
      onDragEnter={(e) => { e.preventDefault(); setDragActive(true) }}
      onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
      onDragLeave={(e) => { e.preventDefault(); setDragActive(false) }}
      onDrop={(e) => {
        e.preventDefault()
        setDragActive(false)
        handleFile(e.dataTransfer.files?.[0])
      }}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ALLOWED_TYPES.join(',')}
        hidden
        onChange={(e) => {
          handleFile(e.target.files?.[0])
          e.target.value = ''   // allows re-selecting the same file
        }}
      />
      <div className="upload-icon">🔍</div>
      <h2>Drop an image here</h2>
      <p>or click to browse</p>
      <p className="upload-hint">JPEG · PNG · WEBP · BMP — max 10 MB</p>
    </div>
  )
}