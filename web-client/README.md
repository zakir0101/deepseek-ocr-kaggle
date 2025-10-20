# DeepSeek OCR React Client

A modern React application for interacting with the DeepSeek OCR Flask server.

## Features

- **Drag & Drop Interface**: Easy image upload with drag and drop support
- **Real-time Preview**: Preview images before processing
- **Markdown Rendering**: View OCR results as rendered markdown
- **Source View**: Toggle between rendered markdown and source text
- **Bounding Box Visualization**: See detected regions with bounding boxes
- **Extracted Images Gallery**: View all extracted sub-images
- **Responsive Design**: Works on desktop and mobile devices

## Setup

### Prerequisites

- Node.js 16+ and npm
- DeepSeek OCR Flask server running on `http://localhost:5000`

### Installation

1. **Navigate to the web-client directory:**
   ```bash
   cd DeepSeek-OCR-vllm/web-client
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser to:**
   ```
   http://localhost:3000
   ```

## Usage

1. **Upload Image**: Drag and drop an image file or click to select
2. **Preview**: See the uploaded image before processing
3. **Process**: Click "Extract Text" to send to the OCR server
4. **View Results**:
   - See the original image and image with bounding boxes
   - Toggle between rendered markdown and source view
   - Browse extracted sub-images
5. **Process Another**: Click "Process Another Image" to start over

## API Integration

The client communicates with the Flask server through:

- **POST /api/ocr/image**: Send image for OCR processing
- **GET /images/{filename}**: Retrieve extracted images
- **Proxy Configuration**: Vite dev server proxies API calls to Flask

## Project Structure

```
web-client/
├── src/
│   ├── App.jsx              # Main application component
│   ├── main.jsx             # React entry point
│   ├── index.css            # Global styles
│   └── components/          # Reusable components (future)
├── package.json             # Dependencies and scripts
├── vite.config.js           # Vite configuration
└── index.html              # HTML template
```

## Dependencies

### Core Dependencies
- **React 18**: UI framework
- **React DOM**: DOM rendering
- **React Markdown**: Markdown rendering
- **Axios**: HTTP client for API calls
- **Lucide React**: Icon library

### Development Dependencies
- **Vite**: Build tool and dev server
- **@vitejs/plugin-react**: React plugin for Vite

## Customization

### Styling
- CSS is organized in `index.css`
- Uses a clean, modern design with responsive layout
- Easy to customize colors, fonts, and layout

### Components
- Main application logic is in `App.jsx`
- Can be extended with additional components in `src/components/`

### API Configuration
- Server URL is configured in `vite.config.js`
- Change the proxy target if server runs on different port

## Build for Production

```bash
npm run build
```

This creates a `dist` folder with optimized production files.

## Troubleshooting

1. **Connection Errors**: Ensure Flask server is running on port 5000
2. **CORS Issues**: Check Flask server CORS configuration
3. **Build Errors**: Clear node_modules and reinstall dependencies
4. **Image Upload Issues**: Verify image format and size

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

Part of the DeepSeek OCR project. See main project for license details.