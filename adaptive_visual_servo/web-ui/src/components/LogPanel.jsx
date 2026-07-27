import { useEffect, useRef } from 'react';

export default function LogPanel({ logs, onClear }) {
  const containerRef = useRef(null);

  useEffect(() => {
    const el = containerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs]);

  return (
    <div className="card card-log">
      <div className="card-header">
        <h2>📋 Log</h2>
        <button className="btn btn-sm btn-ghost" onClick={onClear}>Clear</button>
      </div>
      <div className="log-container" ref={containerRef}>
        {logs.map((l, i) => (
          <div key={i} className={`log-line ${l.cls}`}>{l.msg}</div>
        ))}
      </div>
    </div>
  );
}
