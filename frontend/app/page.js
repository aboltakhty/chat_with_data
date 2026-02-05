'use client';

import { useMemo, useState } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [file, setFile] = useState(null);
  const [ingestStatus, setIngestStatus] = useState('idle');
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState('');
  const [isThinking, setIsThinking] = useState(false);

  const [temperature, setTemperature] = useState(0.2);
  const [topK, setTopK] = useState(4);
  const [answerStyle, setAnswerStyle] = useState('precise');
  const [modelOverride, setModelOverride] = useState('');
  const [showAnalysis, setShowAnalysis] = useState(true);
  const [showChecker, setShowChecker] = useState(true);

  const canChat = useMemo(() => ingestStatus === 'ready', [ingestStatus]);

  const formatMessage = (text = '') => {
    const escaped = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\"/g, '&quot;')
      .replace(/'/g, '&#039;');

    let html = escaped.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/(^|[^*])\*([^*]+)\*/g, '$1<em>$2</em>');
    html = html.replace(/\n/g, '<br />');
    return { __html: html };
  };

  const handleUpload = async () => {
    if (!file) return;
    setError('');
    setIngestStatus('loading');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_URL}/ingest`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || 'Failed to ingest PDF');
      }
      const data = await res.json();
      setIngestStatus('ready');
      setMessages((prev) => [
        ...prev,
        { role: 'system', content: `Loaded: ${data.filename} (${data.chunks} chunks)` }
      ]);
    } catch (err) {
      setIngestStatus('idle');
      setError(err.message || 'Upload failed');
    }
  };

  const handleAsk = async () => {
    if (!question.trim() || !canChat) return;
    setError('');
    setIsThinking(true);

    const buildId = () =>
      (globalThis.crypto?.randomUUID?.() ||
        `${Date.now()}-${Math.random().toString(16).slice(2)}`);

    const userMessage = { id: buildId(), role: 'user', content: question.trim() };
    const assistantId = buildId();
    const assistantMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      reasoning: '',
      status: 'streaming'
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setQuestion('');

    const updateAssistant = (patch) => {
      setMessages((prev) =>
        prev.map((msg) => (msg.id === assistantId ? { ...msg, ...patch } : msg))
      );
    };

    const appendAssistant = (field, delta) => {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? { ...msg, [field]: `${msg[field] || ''}${delta}` }
            : msg
        )
      );
    };

    try {
      const payload = {
        question: userMessage.content,
        temperature,
        top_k: topK,
        answer_style: answerStyle,
        reasoning_effort: 'medium'
      };
      if (modelOverride.trim()) {
        payload.model = modelOverride.trim();
      }

      const res = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok || !res.body) {
        const text = await res.text();
        throw new Error(text || 'Streaming failed');
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split('\\n\\n');
        buffer = events.pop() || '';

        for (const eventChunk of events) {
          const lines = eventChunk.split('\\n');
          let eventName = 'message';
          const dataLines = [];
          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventName = line.replace('event:', '').trim();
            } else if (line.startsWith('data:')) {
              dataLines.push(line.replace('data:', '').trim());
            }
          }
          const dataStr = dataLines.join('\\n');
          let data = {};
          try {
            data = dataStr ? JSON.parse(dataStr) : {};
          } catch {
            data = { delta: dataStr };
          }

          if (eventName === 'response.output_text.delta') {
            appendAssistant('content', data.delta || '');
          } else if (
            eventName === 'response.reasoning_text.delta' ||
            eventName === 'response.reasoning_summary_text.delta'
          ) {
            appendAssistant('reasoning', data.delta || '');
          } else if (eventName === 'response.completed') {
            updateAssistant({ status: 'done' });
          } else if (eventName === 'error') {
            throw new Error(data.message || 'Streaming error');
          }
        }
      }
    } catch (err) {
      updateAssistant({ status: 'error' });
      setError(err.message || 'Chat failed');
    } finally {
      updateAssistant({ status: 'done' });
      setIsThinking(false);
    }
  };

  return (
    <main className="page">
      <header className="topbar">
        <div>
          <p className="eyebrow">PDF Intelligence Suite</p>
          <h1>Chat with PDFs in a cinematic workspace.</h1>
          <p className="subhead">
            Upload a file, tune the model, and get precise answers grounded in your document.
          </p>
        </div>
        <div className="status-pill">
          <span className={canChat ? 'status ready' : 'status'}>
            {canChat ? 'Index ready' : 'Awaiting PDF'}
          </span>
          <span className="pill">Style: {answerStyle}</span>
        </div>
      </header>

      <section className="layout">
        <section className="left-pane">
          <div className="chat-panel">
            <div className="chat-header">
              <h2>Conversation</h2>
              <span className="hint">Answers are tuned for precision and clarity.</span>
            </div>
            <div className="chat-stream">
              {messages.length === 0 && (
                <div className="empty-state">
                  <p>No messages yet. Upload a PDF and ask your first question.</p>
                </div>
              )}
              {messages.map((msg, idx) => (
                <div key={idx} className={`bubble ${msg.role}`}>
                  <p className="role">{msg.role.toUpperCase()}</p>
                  {msg.role === 'assistant' ? (
                    <div className="answer-card">
                      <p className="answer-title">Answer</p>
                      <p className="content" dangerouslySetInnerHTML={formatMessage(msg.content)} />
                      {(msg.status === 'streaming' || msg.reasoning) && (
                        <details className="thinking-drawer" open={msg.status === 'streaming'}>
                          <summary>
                            <span>Thinking</span>
                            {msg.status === 'streaming' && <span className="live-dot">Live</span>}
                          </summary>
                          <div className="thinking-body">
                            <div className="walking-person" aria-hidden="true">
                              <svg className="walker" viewBox="0 0 64 64">
                                <circle cx="32" cy="14" r="6" />
                                <line x1="32" y1="20" x2="32" y2="36" />
                                <line x1="32" y1="26" x2="22" y2="32" />
                                <line x1="32" y1="26" x2="42" y2="32" />
                                <line x1="32" y1="36" x2="24" y2="50" />
                                <line x1="32" y1="36" x2="40" y2="50" />
                              </svg>
                              <div className="thought-bubble">
                                <span />
                                <span />
                                <span />
                              </div>
                            </div>
                            <pre className="thinking-text">
                              {msg.reasoning || 'Waiting for reasoning stream…'}
                            </pre>
                          </div>
                        </details>
                      )}
                      <div className="meta">
                        {showAnalysis && msg.analysis && (
                          <details>
                            <summary>Agent analysis</summary>
                            <pre>{msg.analysis}</pre>
                          </details>
                        )}
                        {showChecker && msg.checker && (
                          <details>
                            <summary>Agent checker</summary>
                            <pre>{msg.checker}</pre>
                          </details>
                        )}
                      </div>
                    </div>
                  ) : (
                    <p className="content" dangerouslySetInnerHTML={formatMessage(msg.content)} />
                  )}
                </div>
              ))}
            </div>
            <div className="composer">
              <textarea
                rows={3}
                placeholder="Ask a question about the loaded PDF…"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                disabled={!canChat || isThinking}
              />
              <button
                className="primary"
                onClick={handleAsk}
                disabled={!canChat || isThinking || !question.trim()}
              >
                Send
              </button>
            </div>
            {error && <p className="error">{error}</p>}
          </div>
        </section>

        <aside className="right-pane">
          <div className="panel-card">
            <h3>Upload PDF</h3>
            <label className="file-label">
              <span>{file ? file.name : 'Choose a PDF'}</span>
              <input
                type="file"
                accept="application/pdf"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
            </label>
            <button className="primary" onClick={handleUpload} disabled={!file || ingestStatus === 'loading'}>
              {ingestStatus === 'loading' ? 'Indexing…' : 'Load PDF'}
            </button>
            <p className="hint">Embeddings + FAISS index run in the backend.</p>
          </div>

          <div className="panel-card">
            <h3>Model Controls</h3>
            <div className="control">
              <label>Answer style</label>
              <select value={answerStyle} onChange={(e) => setAnswerStyle(e.target.value)}>
                <option value="precise">Precise</option>
                <option value="balanced">Balanced</option>
                <option value="detailed">Detailed</option>
              </select>
            </div>
            <div className="control">
              <label>Temperature: {temperature.toFixed(2)}</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
              />
            </div>
            <div className="control">
              <label>Context depth: {topK}</label>
              <input
                type="range"
                min="2"
                max="8"
                step="1"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value, 10))}
              />
            </div>
            <div className="control">
              <label>Model override (optional)</label>
              <input
                type="text"
                placeholder="gpt-5.1"
                value={modelOverride}
                onChange={(e) => setModelOverride(e.target.value)}
              />
            </div>
          </div>

          <div className="panel-card">
            <h3>Agent Visibility</h3>
            <label className="toggle">
              <input
                type="checkbox"
                checked={showAnalysis}
                onChange={(e) => setShowAnalysis(e.target.checked)}
              />
              Show analysis
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={showChecker}
                onChange={(e) => setShowChecker(e.target.checked)}
              />
              Show checker
            </label>
          </div>
        </aside>
      </section>
    </main>
  );
}
