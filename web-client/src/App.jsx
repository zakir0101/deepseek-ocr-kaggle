import React, { useState, useCallback, useEffect } from 'react'
import { Upload, FileText, Image as ImageIcon, Loader, Server, Wifi, WifiOff } from 'lucide-react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import config, { getApiUrl } from './config'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState('rendered')
  const [isDragOver, setIsDragOver] = useState(false)
  const [serverStatus, setServerStatus] = useState('checking')

  const handleFileSelect = useCallback((file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file)
      setError('')

      // Create preview URL
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
    } else {
      setError('Please select a valid image file (JPEG, PNG, etc.)')
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)

    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileSelect(files[0])
    }
  }, [handleFileSelect])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleFileInput = useCallback((e) => {
    const file = e.target.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }, [handleFileSelect])

  // Check server status on component mount
  useEffect(() => {
    checkServerStatus()
  }, [])

  // Process MathJax equations when result changes
  useEffect(() => {
    if (result && activeTab === 'rendered' && window.MathJax) {
      // Give the DOM time to update, then process MathJax
      setTimeout(() => {
        window.MathJax.typesetPromise && window.MathJax.typesetPromise();
      }, 100)
    }
  }, [result, activeTab])

  const checkServerStatus = async () => {
    try {
      const healthUrl = getApiUrl('health')
      console.log('Checking server status at:', healthUrl)

      const response = await axios.get(healthUrl, {
        timeout: 10000, // 10 seconds
        validateStatus: function (status) {
          return status < 500; // Resolve only if the status code is less than 500
        }
      })

      console.log('Server response:', response.data)

      if (response.data.status === 'healthy') {
        setServerStatus('connected')
      } else {
        setServerStatus('error')
      }
    } catch (err) {
      console.error('Server connection failed:', err.message)
      console.error('Error details:', err.response?.data || err.code)
      setServerStatus('disconnected')
    }
  }


  const processOCR = async () => {
    if (!selectedFile) {
      setError('Please select an image first')
      return
    }

    if (serverStatus !== 'connected') {
      setError('Cannot connect to OCR server. Please check server status.')
      return
    }

    setIsLoading(true)
    setError('')
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('image', selectedFile)

      const response = await axios.post(getApiUrl('ocrImage'), formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 60000 // 60 second timeout for OCR processing
      })

      if (response.data.success) {
        setResult(response.data)
      } else {
        setError(response.data.error || 'OCR processing failed')
      }
    } catch (err) {
      if (err.code === 'ECONNABORTED') {
        setError('Request timeout. The server might be processing a large image.')
      } else {
        setError(err.response?.data?.error || 'Failed to process image. Please try again.')
      }
    } finally {
      setIsLoading(false)
    }
  }

  const resetForm = () => {
    setSelectedFile(null)
    setPreviewUrl('')
    setResult(null)
    setError('')
    setActiveTab('rendered')

    // Clean up preview URL
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }
  }

  return (
    <div className="container">
      <div className="header">
        <h1>DeepSeek OCR</h1>
        <p>Upload an image to extract text and convert to markdown</p>

        {/* Server Status Indicator */}
        <div className="server-status">
          <div className={`status-indicator ${serverStatus}`}>
            {serverStatus === 'connected' && <Wifi size={16} />}
            {serverStatus === 'disconnected' && <WifiOff size={16} />}
            {serverStatus === 'checking' && <Loader size={16} />}
            <span>
              Server: {serverStatus === 'connected' ? 'Connected' :
                serverStatus === 'disconnected' ? 'Disconnected' :
                  'Checking...'}
            </span>
            <button
              onClick={checkServerStatus}
              style={{
                marginLeft: '10px',
                padding: '2px 8px',
                fontSize: '12px',
                background: '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '3px',
                cursor: 'pointer'
              }}
            >
              Refresh
            </button>
          </div>


          {result?.demo_mode && (
            <div className="demo-notice">
              ⚠ Running in demo mode - install vLLM for full functionality
            </div>
          )}
        </div>

      </div>

      {!result && (
        <div className="upload-section">
          <div
            className={`upload-area ${isDragOver ? 'drag-over' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => document.getElementById('file-input').click()}
          >
            <Upload className="upload-icon" />
            <h3>Drag & Drop Image Here</h3>
            <p>or click to select a file</p>
            <p style={{ fontSize: '14px', color: '#666', marginTop: '10px' }}>
              Supported formats: JPEG, PNG, WebP
            </p>
          </div>

          <input
            id="file-input"
            type="file"
            accept="image/*"
            onChange={handleFileInput}
            style={{ display: 'none' }}
          />

          {previewUrl && (
            <div className="preview-container">
              <h4>Preview:</h4>
              <img
                src={previewUrl}
                alt="Preview"
                className="preview-image-upload"
              />
              <button
                className="button"
                onClick={processOCR}
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader size={16} style={{ marginRight: '8px' }} />
                    Processing...
                  </>
                ) : (
                  'Extract Text'
                )}
              </button>
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="error">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <>
          <div className="preview-section">
            <div className="preview-image">
              <h3>
                <ImageIcon size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                Original Image
              </h3>
              <img src={previewUrl} alt="Original" />
            </div>

            {result.boxes_image && (
              <div className="preview-image">
                <h3>
                  <FileText size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                  Image with Bounding Boxes
                </h3>
                <img src={`data:image/jpeg;base64,${result.boxes_image}`} alt="With bounding boxes" />
              </div>
            )}
          </div>

          <div className="result-section">
            <div className="result-tabs">
              <button
                className={`tab ${activeTab === 'rendered' ? 'active' : ''}`}
                onClick={() => setActiveTab('rendered')}
              >
                Rendered Markdown
              </button>
              <button
                className={`tab ${activeTab === 'source' ? 'active' : ''}`}
                onClick={() => setActiveTab('source')}
              >
                Source Markdown
              </button>
              <button
                className={`tab ${activeTab === 'raw' ? 'active' : ''}`}
                onClick={() => setActiveTab('raw')}
              >
                Raw OCR Output
              </button>
            </div>

            {activeTab === 'rendered' ? (
              <div
                className="markdown-content"
                dangerouslySetInnerHTML={{ __html: result.source_markdown }}
              />
            ) : activeTab === 'source' ? (
              <div className="markdown-source">
                {result.markdown}
              </div>
            ) : (
              <div className="markdown-raw">
                <pre>{result.raw_result}</pre>
              </div>
            )}


            <div style={{ textAlign: 'center', marginTop: '20px' }}>
              <button className="button" onClick={resetForm}>
                Process Another Image
              </button>
            </div>
          </div>
        </>
      )}

      {isLoading && (
        <div className="loading">
          <Loader size={48} style={{ marginBottom: '15px' }} />
          <p>Processing image with DeepSeek OCR...</p>
          <p style={{ fontSize: '14px', color: '#666' }}>
            This may take a few moments depending on image size and complexity.
          </p>
        </div>
      )}
    </div>
  )
}

export default App