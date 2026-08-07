import { useState, useEffect } from 'react';
import { api } from '../hooks/useSocket';

export default function ControlPanel({
  armConnected, episodeRunning, speed, probes, trackingMode, pickPx, dropPx, addLog
}) {
  const [loading, setLoading] = useState(null);

  const busy = !armConnected || episodeRunning;

  // Sync loading state with global episode execution state
  useEffect(() => {
    if (!episodeRunning && loading === 'episode') {
      setLoading(null);
    }
  }, [episodeRunning, loading]);

  async function action(name, endpoint, body = {}, timeout = 3000) {
    setLoading(name);
    try {
      await api(endpoint, 'POST', body);
    } catch (e) {
      addLog(`${name} failed: ${e.message}`, 'error');
      setLoading(null);
      return;
    }
    if (name !== 'episode') {
      setTimeout(() => setLoading(null), timeout);
    }
  }

  return (
    <div className="card">
      <div className="card-header"><h2>▶️ Control</h2></div>
      <div className="control-grid">
        <button
          className={`btn btn-primary ${loading === 'episode' ? 'loading' : ''}`}
          disabled={busy || !!loading}
          onClick={() => action('episode', 'run_episode', {
            speed, probes, tracking_mode: trackingMode,
            pick_px: pickPx, drop_px: dropPx,
          }, 0)}
        >Run Episode</button>
        <button
          className={`btn ${loading === 'bg' ? 'loading' : ''}`}
          disabled={busy || !!loading}
          onClick={() => action('bg', 'capture_background', {}, 1500)}
        >Capture BG</button>
        <button
          className={`btn ${loading === 'test' ? 'loading' : ''}`}
          disabled={busy || !!loading}
          onClick={() => action('test', 'test_tracking', { tracking_mode: trackingMode }, 2000)}
        >Test Tracking</button>
        <button
          className={`btn btn-danger ${loading === 'reset' ? 'loading' : ''}`}
          disabled={busy || !!loading}
          onClick={() => action('reset', 'reset_arm', {}, 4000)}
        >Reset Arm</button>
      </div>
      <div className="control-grid" style={{ marginTop: 8 }}>
        <button
          className={`btn btn-accent ${loading === 'circle' ? 'loading' : ''}`}
          disabled={busy || !!loading}
          onClick={() => action('circle', 'draw_shape', { shape: 'circle' }, 10000)}
        >⭕ Circle</button>
        <button
          className={`btn btn-accent ${loading === 'heart' ? 'loading' : ''}`}
          disabled={busy || !!loading}
          onClick={() => action('heart', 'draw_shape', { shape: 'heart' }, 10000)}
        >❤️ Heart</button>
      </div>
    </div>
  );
}
