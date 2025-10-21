// Configuration for DeepSeek OCR Client
// This file allows easy configuration for different deployment environments

const config = {
  // Server configuration
  server: {
    // Default server URL (for local development)
    local: 'http://localhost:5000',

    // For Kaggle deployment, this will be the ngrok URL
    // You can set this dynamically based on the environment
    remote: 'https://301bc9609d7f.ngrok-free.app',

    // Auto-detect which server to use
    getBaseUrl: function() {
      // If we have a remote URL configured, use it
      if (this.remote) {
        return this.remote;
      }

      // Otherwise use local server (for development)
      return this.local;
    },

    // API endpoints
    endpoints: {
      ocrImage: '/api/ocr/image',
      ocrPdf: '/api/ocr/pdf',
      health: '/health',
      images: '/images'
    }
  },

  // Application settings
  app: {
    name: 'DeepSeek OCR',
    version: '1.0.0',
    description: 'OCR with DeepSeek AI models',

    // File upload settings
    upload: {
      maxFileSize: 10 * 1024 * 1024, // 10MB
      allowedTypes: ['image/jpeg', 'image/png', 'image/webp', 'image/gif'],

      // Validation messages
      messages: {
        fileTooLarge: 'File size must be less than 10MB',
        invalidType: 'Please upload a valid image file (JPEG, PNG, WebP, GIF)',
        uploadError: 'Failed to upload file. Please try again.'
      }
    },

    // OCR processing settings
    processing: {
      timeout: 30000, // 30 seconds
      retryAttempts: 3,

      // Default prompts
      prompts: {
        document: '<image>\n<|grounding|>Convert the document to markdown.',
        freeOcr: '<image>\nFree OCR.',
        parseFigure: '<image>\nParse the figure.',
        describe: '<image>\nDescribe this image in detail.'
      }
    }
  },

  // UI settings
  ui: {
    theme: {
      primary: '#007bff',
      secondary: '#6c757d',
      success: '#28a745',
      danger: '#dc3545',
      warning: '#ffc107',
      info: '#17a2b8'
    },

    layout: {
      maxWidth: '1200px',
      sidebarWidth: '300px',
      headerHeight: '80px'
    }
  }
};

// Server URL helper
export const getServerUrl = () => {
  // Always use the configured server URL
  return config.server.getBaseUrl();
};

// API URL helper
export const getApiUrl = (endpoint) => {
  const baseUrl = getServerUrl();
  const fullEndpoint = config.server.endpoints[endpoint] || endpoint;

  // If we have a base URL, prepend it
  if (baseUrl) {
    return `${baseUrl}${fullEndpoint}`;
  }

  // Otherwise use relative URL (will be proxied in dev)
  return fullEndpoint;
};

// Development mode check
export const isDevelopment = () => {
  return window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
};

export default config;
