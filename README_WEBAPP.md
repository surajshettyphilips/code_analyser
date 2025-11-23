# PySpark Code Analyzer - Web Application

Full-stack web application for analyzing PySpark code using ChromaDB, DSPy, and CodeLlama.

## Architecture

### Backend (FastAPI)
- **Location**: `api/main.py`
- **Port**: 8000
- **Endpoints**:
  - `POST /upload` - Upload and index code files (Stage 1)
  - `POST /query` - Query indexed code (Stage 2)
  - `GET /collection/info` - Get collection statistics
  - `DELETE /collection/reset` - Clear all indexed data

### Frontend (React + Vite)
- **Location**: `frontend/`
- **Port**: 3000
- **Features**:
  - File upload with drag-and-drop
  - Natural language question interface
  - Real-time answer display with code contexts
  - Collection statistics

## Setup Instructions

### 1. Install Backend Dependencies

```powershell
.\venv\Scripts\activate
pip install fastapi uvicorn python-multipart
```

### 2. Install Frontend Dependencies

```powershell
cd frontend
npm install
```

### 3. Start the Backend Server

```powershell
# From project root
.\venv\Scripts\activate
python api/main.py
```

The API will be available at: http://localhost:8000

### 4. Start the Frontend Development Server

```powershell
# In a new terminal
cd frontend
npm run dev
```

The web app will be available at: http://localhost:3000

## Usage

### Step 1: Upload Code File
1. Open http://localhost:3000
2. Drag & drop a Python file or click to browse
3. Click "Upload & Index" button
4. Wait for confirmation message

### Step 2: Ask Questions
1. Type your question in the text box
   - Example: "What business logic is implemented in this code?"
   - Example: "How does this code handle data validation?"
2. Click "Search" button
3. View the AI-generated answer and relevant code contexts below

## API Examples

### Upload File
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@example_pyspark_etl.py"
```

### Query Code
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main business rules?", "n_results": 5}'
```

### Get Collection Info
```bash
curl "http://localhost:8000/collection/info"
```

## Technology Stack

**Backend:**
- FastAPI - Modern Python web framework
- ChromaDB - Vector database for code storage
- DSPy - LLM orchestration
- CodeLlama - Code analysis model via Ollama

**Frontend:**
- React 18 - UI library
- Vite - Build tool and dev server
- Axios - HTTP client
- CSS3 - Styling with animations

## Project Structure

```
.
├── api/
│   └── main.py              # FastAPI backend server
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUpload.jsx    # File upload component
│   │   │   ├── FileUpload.css
│   │   │   ├── QuerySection.jsx  # Query interface component
│   │   │   └── QuerySection.css
│   │   ├── App.jsx          # Main app component
│   │   ├── App.css
│   │   ├── main.jsx         # Entry point
│   │   └── index.css        # Global styles
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── README_WEBAPP.md         # This file
```

## Features

✅ **File Upload**
- Drag & drop interface
- File validation (.py, .txt)
- Upload progress indicator
- Success/error feedback

✅ **Question Answering**
- Natural language queries
- Real-time search
- Formatted answers with code contexts
- Source file tracking
- Distance metrics for relevance

✅ **User Experience**
- Responsive design
- Loading states
- Error handling
- Smooth animations
- Collection statistics

## Notes

- Ensure Ollama is running with CodeLlama model before starting
- Backend must be running before frontend for API calls to work
- ChromaDB data persists in `c:\Users\320196443\codellama\chromadb`
- The frontend uses a proxy to forward `/api/*` requests to backend
