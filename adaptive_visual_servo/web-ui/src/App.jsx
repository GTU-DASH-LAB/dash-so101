import { useState, useEffect, useRef, useCallback } from 'react';
import { useSocket, api } from './hooks/useSocket';
import VideoFeed from './components/VideoFeed';
import ConnectionPanel from './components/ConnectionPanel';
import CalibrationDashboard from './components/CalibrationDashboard';
import ControlPanel from './components/ControlPanel';
import SpeedTuning from './components/SpeedTuning';
import LogPanel from './components/LogPanel';

export default function App() {
  const { on, connected: wsConnected } = useSocket();
  const [armConnected, setArmConnected] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [frameData, setFrameData] = useState(null);
  const [markers, setMarkers] = useState({});
  const [logs, setLogs] = useState([{ msg: 'System ready. Scan cameras to begin.', cls: 'system' }]);
  const [status, setStatus] = useState({});
  const [episodeRunning, setEpisodeRunning] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [probes, setProbes] = useState(22);
  const [trackingMode, setTrackingMode] = useState('aruco');
  const [pickPx, setPickPx] = useState(null);
  const [dropPx, setDropPx] = useState(null);
  const [clickMode, setClickMode] = useState(null);

  // FPS
  const frameCountRef = useRef(0);
  const [fps, setFps] = useState(0);

  const addLog = useCallback((msg, cls = '') => {
    const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
    setLogs(prev => {
      const next = [...prev, { msg: `[${ts}] ${msg}`, cls }];
      return next.length > 200 ? next.slice(-200) : next;
    });
  }, []);

  // Socket events
  useEffect(() => {
    const unsub1 = on('frame', (data) => {
      setFrameData(data.image);
      setMarkers(data.markers || {});
      frameCountRef.current++;
    });
    const unsub2 = on('log', (data) => {
      const msg = data.msg || '';
      const cls = msg.toLowerCase().includes('fail') || msg.toLowerCase().includes('error')
        ? 'error'
        : msg.toLowerCase().includes('complete') || msg.toLowerCase().includes('done')
        ? 'success' : '';
      addLog(msg, cls);
    });
    return () => { unsub1(); unsub2(); };
  }, [on, addLog]);

  // FPS counter
  useEffect(() => {
    const interval = setInterval(() => {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Periodic status poll
  useEffect(() => {
    const poll = async () => {
      try {
        const s = await api('status');
        setStatus(s);
        setArmConnected(s.connected);
        setEpisodeRunning(s.episode_running);
      } catch (_) {}
    };
    poll();
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <>
      <header className="app-header">
        <div className="header-left">
          <div className="logo-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z"/>
              <path d="M2 17l10 5 10-5"/>
              <path d="M2 12l10 5 10-5"/>
            </svg>
          </div>
          <h1>SO-101 Visual Servo</h1>
          <span className={`badge ${armConnected ? 'connected' : ''}`}>
            {armConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        <div className="header-right" style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          {armConnected && (
            <button
              onClick={async () => {
                try {
                  await api('estop', 'POST');
                } catch (e) {
                  addLog('Failed to send E-STOP: ' + e.message, 'error');
                }
              }}
              style={{
                backgroundColor: '#dc2626',
                color: 'white',
                border: 'none',
                padding: '6px 16px',
                borderRadius: '6px',
                cursor: 'pointer',
                fontWeight: 'bold',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                boxShadow: '0 0 10px rgba(220, 38, 38, 0.5)',
                transition: 'all 0.2s',
              }}
              onMouseOver={e => { e.currentTarget.style.backgroundColor = '#ef4444'; e.currentTarget.style.boxShadow = '0 0 15px rgba(239, 68, 68, 0.8)'; }}
              onMouseOut={e => { e.currentTarget.style.backgroundColor = '#dc2626'; e.currentTarget.style.boxShadow = '0 0 10px rgba(220, 38, 38, 0.5)'; }}
            >🚨 E-Stop</button>
          )}
          <span className="header-info">{fps > 0 ? `${fps} fps` : '—'}</span>
        </div>
      </header>

      <main className="app-main">
        <section className="col-left">
          <VideoFeed
            frameData={frameData}
            markers={markers}
            pickPx={pickPx}
            dropPx={dropPx}
            clickMode={clickMode}
            onClickCanvas={(mode, px) => {
              if (mode === 'pick') setPickPx(px);
              else setDropPx(px);
              setClickMode(null);
              addLog(`${mode.toUpperCase()} set: [${px[0].toFixed(0)}, ${px[1].toFixed(0)}]`, 'system');
            }}
            onSetPick={() => { setClickMode('pick'); addLog('Click the video to set PICK target.', 'system'); }}
            onSetDrop={() => { setClickMode('drop'); addLog('Click the video to set DROP location.', 'system'); }}
            onClear={() => { setPickPx(null); setDropPx(null); setClickMode(null); }}
          />

          <ControlPanel
            armConnected={armConnected}
            episodeRunning={episodeRunning}
            speed={speed}
            probes={probes}
            trackingMode={trackingMode}
            pickPx={pickPx}
            dropPx={dropPx}
            addLog={addLog}
          />

          <SpeedTuning
            speed={speed}
            setSpeed={setSpeed}
            probes={probes}
            setProbes={setProbes}
          />
        </section>

        <section className="col-right">
          <ConnectionPanel
            armConnected={armConnected}
            setArmConnected={setArmConnected}
            previewing={previewing}
            setPreviewing={setPreviewing}
            addLog={addLog}
          />

          <CalibrationDashboard
            markers={markers}
            status={status}
            armConnected={armConnected}
            previewing={previewing}
            trackingMode={trackingMode}
            setTrackingMode={setTrackingMode}
            addLog={addLog}
          />

          <LogPanel logs={logs} onClear={() => setLogs([])} />
        </section>
      </main>
    </>
  );
}
