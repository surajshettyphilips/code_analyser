import React, { useState } from 'react';
import './EvalSection.css';

function EvalSection() {
  const [file, setFile] = useState(null);
  const [evaluating, setEvaluating] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [showResults, setShowResults] = useState(true);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (!selectedFile.name.endsWith('.xlsx') && !selectedFile.name.endsWith('.xls')) {
        setError('Only Excel files (.xlsx, .xls) are supported');
        setFile(null);
        return;
      }
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleEvaluate = async () => {
    if (!file) {
      setError('Please select an Excel file first');
      return;
    }

    setEvaluating(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/eval', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Evaluation failed');
      }

      const data = await response.json();
      setResults(data);
      setShowResults(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setEvaluating(false);
    }
  };

  const getMetricColor = (value) => {
    if (value >= 0.7) return '#28a745';
    if (value >= 0.5) return '#ffc107';
    return '#dc3545';
  };

  const formatMetric = (value) => {
    if (value === null || value === undefined || value === 'N/A') return 'N/A';
    return typeof value === 'number' ? value.toFixed(4) : value;
  };

  return (
    <div className="eval-section">
      <div className="eval-upload">
        <div className="file-input-wrapper">
          <input
            type="file"
            id="eval-file-input"
            accept=".xlsx,.xls"
            onChange={handleFileChange}
            className="file-input"
          />
          <label htmlFor="eval-file-input" className="file-input-label">
            {file ? (
              <>
                📊 {file.name}
                <span className="file-size"> ({(file.size / 1024).toFixed(2)} KB)</span>
              </>
            ) : (
              <>📊 Choose Excel File (questions + ground_truth)</>
            )}
          </label>
        </div>

        <button
          className="eval-button"
          onClick={handleEvaluate}
          disabled={!file || evaluating}
        >
          {evaluating ? (
            <>
              <span className="spinner"></span> Evaluating...
            </>
          ) : (
            <>🎯 Evaluate with RAGAs</>
          )}
        </button>
      </div>

      {error && (
        <div className="error-message">
          ❌ <strong>Error:</strong> {error}
        </div>
      )}

      {results && (
        <div className="eval-results">
          <div className="results-header">
            <div className="results-info">
              <h3>📈 Evaluation Results</h3>
              <span className="results-count">
                {results.total_questions} questions evaluated
              </span>
            </div>
            <button
              className="toggle-button"
              onClick={() => setShowResults(!showResults)}
            >
              {showResults ? '🔽 Hide Results' : '🔼 Show Results'}
            </button>
          </div>

          {showResults && (
            <div className="results-table-container">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Question</th>
                    <th>Ground Truth</th>
                    <th>AI Response</th>
                    <th>Contexts</th>
                    <th>Context Precision</th>
                    <th>Context Recall</th>
                    <th>Faithfulness</th>
                    <th>Answer Relevancy</th>
                  </tr>
                </thead>
                <tbody>
                  {results.results.map((row, index) => (
                    <tr key={index}>
                      <td className="index-cell">{index + 1}</td>
                      <td className="question-cell">{row.question}</td>
                      <td className="ground-truth-cell">{row.ground_truth}</td>
                      <td className="response-cell">{row.AI_response || 'N/A'}</td>
                      <td className="contexts-cell">
                        {row.contexts ? (
                          <details>
                            <summary>{Array.isArray(row.contexts) ? row.contexts.length : 1} contexts</summary>
                            <pre>{typeof row.contexts === 'string' ? row.contexts : JSON.stringify(row.contexts, null, 2)}</pre>
                          </details>
                        ) : (
                          'N/A'
                        )}
                      </td>
                      <td className="metric-cell">
                        <span
                          className="metric-value"
                          style={{ color: getMetricColor(row.ragas_context_precision) }}
                        >
                          {formatMetric(row.ragas_context_precision)}
                        </span>
                      </td>
                      <td className="metric-cell">
                        <span
                          className="metric-value"
                          style={{ color: getMetricColor(row.ragas_context_recall) }}
                        >
                          {formatMetric(row.ragas_context_recall)}
                        </span>
                      </td>
                      <td className="metric-cell">
                        <span
                          className="metric-value"
                          style={{ color: getMetricColor(row.ragas_faithfulness) }}
                        >
                          {formatMetric(row.ragas_faithfulness)}
                        </span>
                      </td>
                      <td className="metric-cell">
                        <span
                          className="metric-value"
                          style={{ color: getMetricColor(row.ragas_answer_relevancy) }}
                        >
                          {formatMetric(row.ragas_answer_relevancy)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default EvalSection;
