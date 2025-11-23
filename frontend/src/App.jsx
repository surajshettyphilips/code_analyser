import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import QuerySection from './components/QuerySection';
import EvalSection from './components/EvalSection';
import './App.css';

function App() {
  const [uploadStatus, setUploadStatus] = useState(null);
  const [collectionInfo, setCollectionInfo] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);

  const handleUploadSuccess = (response) => {
    setUploadStatus(response);
    // Add to uploaded files list
    setUploadedFiles(prev => [...prev, {
      filename: response.filename,
      chunks: response.chunks_indexed,
      timestamp: new Date().toLocaleString()
    }]);
    // Refresh collection info after successful upload
    fetchCollectionInfo();
  };

  const fetchCollectionInfo = async () => {
    try {
      const response = await fetch('/api/collection/info');
      if (response.ok) {
        const data = await response.json();
        setCollectionInfo(data);
      }
    } catch (error) {
      console.error('Error fetching collection info:', error);
    }
  };

  React.useEffect(() => {
    fetchCollectionInfo();
  }, []);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🔍 PySpark Code Analyzer</h1>
        <p className="subtitle">Upload code files and query with natural language</p>
        {collectionInfo && (
          <div className="collection-info">
            <span>📊 Total Chunks Indexed: {collectionInfo.total_chunks}</span>
          </div>
        )}
      </header>

      <main className="main-content">
        <section className="upload-section">
          <h2>📤 Step 1: Upload Code File</h2>
          <FileUpload onUploadSuccess={handleUploadSuccess} />
          {uploadStatus && (
            <div className="upload-status success">
              ✅ <strong>{uploadStatus.filename}</strong> uploaded successfully!
              <br />
              Indexed {uploadStatus.chunks_indexed} chunks
            </div>
          )}
          
          {uploadedFiles.length > 0 && (
            <div className="uploaded-files-list">
              <h3>📁 Uploaded Files</h3>
              <div className="files-container">
                {uploadedFiles.map((file, index) => (
                  <div key={index} className="file-card">
                    <div className="file-icon">📄</div>
                    <div className="file-details">
                      <div className="file-name">{file.filename}</div>
                      <div className="file-meta">
                        <span className="file-chunks">{file.chunks} chunks</span>
                        <span className="file-timestamp">{file.timestamp}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        <div className="divider"></div>

        <section className="eval-section-wrapper">
          <h2>🎯 Step 3: Evaluate with RAGAs</h2>
          <EvalSection />
        </section>

        <div className="divider"></div>

        <section className="query-section">
          <h2>🔎 Step 4: Ask Questions</h2>
          <QuerySection />
        </section>
      </main>

      <footer className="app-footer">
        <p>Powered by ChromaDB, DSPy & CodeLlama</p>
      </footer>
    </div>
  );
}

export default App;
