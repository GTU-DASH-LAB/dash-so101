import { useState, useEffect } from 'react';
import { api } from '../hooks/useSocket';

export default function ConnectionPanel({
  armConnected, setArmConnected, previewing, setPreviewing, addLog
}) {
  const [cameras, setCameras] = useState([]);
  const [camIndex, setCamIndex] = useState(0);
  const [port, setPort] = useState('/dev/ttyACM0');
  const [loading, setLoading] = useState(null); // 'scan' | 'connect' | 'preview' | 'detect'
  const [detectStep, setDetectStep] = useState(null); // null | 'unplug' | 'success'

  // Load settings on mount to restore last_port
  useEffect(() => {
    async function loadSettings() {
      try {
        const s = await api('settings');
        if (s.port) setPort(s.port);
        if (s.camera_index != null) setCamIndex(s.camera_index);
      } catch (e) {
        console.error('Failed to load settings', e);
      }
    }
    loadSettings();
    scanCameras();
  }, []);

  async function scanCameras() {
    setLoading('scan');
    try {
      const cams = await api('cameras');
      setCameras(cams);
      if (cams.length > 0) setCamIndex(cams[0].index);
      addLog(`Found ${cams.length} camera(s).`, 'system');
    } catch (e) {
      addLog('Camera scan failed: ' + e.message, 'error');
    }
    setLoading(null);
  }

  async function toggleConnect() {
    setLoading('connect');
    try {
      if (armConnected) {
        await api('disconnect', 'POST');
        setArmConnected(false);
        addLog('Disconnected.', 'system');
      } else {
        if (previewing) {
          await api('preview/stop', 'POST');
          setPreviewing(false);
        }
        await api('connect', 'POST', { camera_index: camIndex, port });
        setArmConnected(true);
      }
    } catch (e) {
      addLog('Connect failed: ' + e.message, 'error');
    }
    setLoading(null);
  }

  async function togglePreview() {
    setLoading('preview');
    try {
      if (previewing) {
        await api('preview/stop', 'POST');
        setPreviewing(false);
      } else if (!armConnected) {
        await api('preview/start', 'POST', { camera_index: camIndex });
        setPreviewing(true);
      } else {
        addLog('Already connected — live feed active.', 'system');
      }
    } catch (e) {
      addLog('Preview failed: ' + e.message, 'error');
    }
    setLoading(null);
  }

  async function startPortDetect() {
    setLoading('detect');
    try {
      await api('ports/snapshot', 'POST');
      setDetectStep('unplug');
      addLog('Unplug the arm USB cable now.', 'system');
    } catch (e) {
      addLog('Port detect failed: ' + e.message, 'error');
    }
    setLoading(null);
  }

  async function confirmUnplug() {
    setLoading('detect');
    try {
      const res = await api('ports/diff', 'POST');
      if (res.detected) {
        setPort(res.port);
        setDetectStep('success');
        addLog(`Arm port detected: ${res.port}. Plug cable back in.`, 'success');
      } else {
        addLog('Port detection failed: ' + (res.error || 'No port disappeared'), 'error');
        setDetectStep(null);
      }
    } catch (e) {
      addLog('Port diff failed: ' + e.message, 'error');
      setDetectStep(null);
    }
    setLoading(null);
  }

  return (
    <div className="card">
      <div className="card-header"><h2>🔌 Connection</h2></div>
      
      {detectStep === 'unplug' ? (
        <div style={{ padding: '8px 0', fontSize: '0.8rem', color: 'var(--warning)' }}>
          <p style={{ marginBottom: 8 }}>⚠️ <strong>Step 2:</strong> Now unplug the arm's USB cable from your computer, then click OK.</p>
          <button className="btn btn-sm btn-primary" onClick={confirmUnplug} disabled={loading === 'detect'}>
            OK, Unplugged
          </button>
        </div>
      ) : detectStep === 'success' ? (
        <div style={{ padding: '8px 0', fontSize: '0.8rem', color: 'var(--success)' }}>
          <p style={{ marginBottom: 8 }}>✅ Detected Port: <strong>{port}</strong>. Plug the USB cable back in now!</p>
          <button className="btn btn-sm" onClick={() => setDetectStep(null)}>
            Finish
          </button>
        </div>
      ) : (
        <>
          <div className="form-row">
            <label>Camera</label>
            <select value={camIndex} onChange={e => setCamIndex(parseInt(e.target.value))}>
              {cameras.length === 0 && <option value="">— scan cameras —</option>}
              {cameras.map(c => (
                <option key={c.index} value={c.index}>
                  {c.index}: {c.width}×{c.height}
                </option>
              ))}
            </select>
            <button className={`btn btn-sm ${loading === 'scan' ? 'loading' : ''}`}
                    onClick={scanCameras} disabled={!!loading}>
              Scan
            </button>
          </div>
          <div className="form-row">
            <label>Arm Port</label>
            <input type="text" value={port} onChange={e => setPort(e.target.value)}
                   placeholder="/dev/ttyACM0" />
            <button className={`btn btn-sm ${loading === 'detect' ? 'loading' : ''}`}
                    onClick={startPortDetect} disabled={!!loading}>
              Detect Port
            </button>
          </div>
          <div className="btn-row">
            <button className={`btn btn-primary ${loading === 'connect' ? 'loading' : ''}`}
                    onClick={toggleConnect} disabled={!!loading}>
              {armConnected ? 'Disconnect' : 'Connect Arm'}
            </button>
            <button className={`btn ${loading === 'preview' ? 'loading' : ''}`}
                    onClick={togglePreview} disabled={!!loading || armConnected}>
              {previewing ? 'Stop Preview' : 'Preview Camera'}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
