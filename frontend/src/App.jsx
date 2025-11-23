import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import QuerySection from './components/QuerySection';
import './App.css';

function App() {
  const [uploadStatus, setUploadStatus] = useState(null);
  const [collectionInfo, setCollectionInfo] = useState(null);

  const handleUploadSuccess = (response) => {
    setUploadStatus(response);
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
        </section>

        <div className="divider"></div>

        <section className="query-section">
          <h2>🔎 Step 2: Ask Questions</h2>
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
