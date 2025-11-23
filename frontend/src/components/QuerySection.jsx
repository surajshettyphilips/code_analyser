import React, { useState } from 'react';
import './QuerySection.css';

function QuerySection() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSearch = async () => {
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question.trim(),
          n_results: 5,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Query failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  };

  return (
    <div className="query-container">
      <div className="query-input-section">
        <textarea
          className="query-input"
          placeholder="Ask a question about your code... (e.g., 'What business logic is implemented in this code?')"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyPress={handleKeyPress}
          rows={3}
        />
        <button
          className="search-button"
          onClick={handleSearch}
          disabled={loading || !question.trim()}
        >
          {loading ? (
            <>
              <span className="spinner"></span> Searching...
            </>
          ) : (
            <>🔍 Search</>
          )}
        </button>
      </div>

      {error && (
        <div className="error-message">
          ❌ <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="result-container">
          <div className="answer-section">
            <h3>💡 Answer</h3>
            <div className="answer-content">
              {result.answer}
            </div>
          </div>

          <div className="contexts-section">
            <h3>📄 Relevant Code Contexts ({result.num_contexts})</h3>
            {result.contexts.map((context, index) => (
              <div key={index} className="context-card">
                <div className="context-header">
                  <span className="context-badge">Context {index + 1}</span>
                  {context.metadata?.source_file && (
                    <span className="context-source">
                      📁 {context.metadata.source_file}
                    </span>
                  )}
                  {context.distance !== null && (
                    <span className="context-distance">
                      Distance: {context.distance.toFixed(4)}
                    </span>
                  )}
                </div>
                <pre className="context-code">
                  <code>{context.text}</code>
                </pre>
                {context.metadata?.chunk_type && (
                  <div className="context-meta">
                    Type: {context.metadata.chunk_type}
                    {context.metadata.line_range && 
                      ` | Lines: ${context.metadata.line_range}`
                    }
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default QuerySection;
