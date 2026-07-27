import { useState } from 'react';
import { api } from '../hooks/useSocket';

const WS_MARKERS = [0, 1, 2, 3];
const ROBOT_MARKERS = [48, 49];
const ALL_MARKERS = [...WS_MARKERS, ...ROBOT_MARKERS];
const MARKER_LABELS = { 0: 'WS-0', 1: 'WS-1', 2: 'WS-2', 3: 'WS-3', 48: 'Gripper', 49: 'Wrist' };

export default function CalibrationDashboard({
  markers, status, armConnected, previewing,
  trackingMode, setTrackingMode, addLog
}) {
  const [wsMarkerSize, setWsMarkerSize] = useState(0.033);
  const [robotMarkerSize, setRobotMarkerSize] = useState(0.018);
  const [tableW, setTableW] = useState(0.63);
  const [tableD, setTableD] = useState(0.60);
  const [loading, setLoading] = useState(null);
  const [charucoCount, setCharucoCount] = useState(status.charuco_frames_collected || 0);
  const [intrinsicsInfo, setIntrinsicsInfo] = useState(null);
  const [workspaceInfo, setWorkspaceInfo] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [gripperId, setGripperId] = useState(48);
  const [wristId, setWristId] = useState(49);
  const [graspZ, setGraspZ] = useState(0.018);

  const detectedIds = Object.keys(markers).map(Number);
  const hasIntrinsics = status.has_intrinsics;
  const hasWorkspace = status.has_workspace;

  const calBadge = hasIntrinsics && hasWorkspace
    ? { text: 'Calibrated', cls: 'calibrated' }
    : hasIntrinsics
    ? { text: 'Intrinsics Only', cls: 'partial' }
    : { text: 'Not Calibrated', cls: '' };

  // Load settings on load
  useState(() => {
    async function loadSettings() {
      try {
        const s = await api('settings');
        if (s.aruco_ws_marker_size) setWsMarkerSize(s.aruco_ws_marker_size);
        if (s.aruco_robot_marker_size) setRobotMarkerSize(s.aruco_robot_marker_size);
        if (s.gripper_marker_id) setGripperId(s.gripper_marker_id);
        if (s.wrist_marker_id) setWristId(s.wrist_marker_id);
        if (s.grasp_z !== undefined) setGraspZ(s.grasp_z);
      } catch (e) {
        console.error('Failed to load settings', e);
      }
    }
    loadSettings();
  }, []);

  async function collectCharuco() {
    setLoading('collect');
    try {
      const res = await api('calibrate/collect_frame', 'POST');
      setCharucoCount(res.frames_collected);
      addLog(`ChArUco frame collected (${res.frames_collected} total).`, 'system');
    } catch (e) { addLog('Collect failed: ' + e.message, 'error'); }
    setLoading(null);
  }

  async function calibrateIntrinsics() {
    setLoading('intrinsics');
    try {
      const res = await api('calibrate/intrinsics', 'POST');
      setIntrinsicsInfo({ ok: true, text: `fx=${res.fx.toFixed(1)} fy=${res.fy.toFixed(1)} | RMS: ${res.rms.toFixed(3)}px` });
      addLog('Intrinsic calibration done!', 'success');
    } catch (e) {
      setIntrinsicsInfo({ ok: false, text: e.message });
      addLog('Intrinsic calibration failed: ' + e.message, 'error');
    }
    setLoading(null);
  }

  async function estimateIntrinsics() {
    setLoading('estimate');
    try {
      const res = await api('calibrate/estimate_intrinsics', 'POST');
      setIntrinsicsInfo({ ok: true, text: `Estimated: fx=${res.fx.toFixed(1)} fy=${res.fy.toFixed(1)}` });
      addLog('Intrinsics estimated from resolution.', 'success');
    } catch (e) { addLog('Estimate failed: ' + e.message, 'error'); }
    setLoading(null);
  }

  async function calibrateWorkspace() {
    setLoading('workspace');
    try {
      const positions = {
        0: [0, 0, 0],
        1: [tableW, 0, 0],
        2: [tableW, tableD, 0],
        3: [0, tableD, 0],
      };
      const res = await api('calibrate/workspace', 'POST', {
        marker_positions: positions,
        marker_size: wsMarkerSize,
      });
      setWorkspaceInfo({ ok: true, text: `RMS reprojection: ${res.rms.toFixed(3)}px` });
      addLog('Workspace calibration done!', 'success');
    } catch (e) {
      setWorkspaceInfo({ ok: false, text: e.message });
      addLog('Workspace calibration failed: ' + e.message, 'error');
    }
    setLoading(null);
  }

  async function resetCalibration() {
    try {
      await api('calibrate/reset', 'POST');
      setIntrinsicsInfo(null);
      setWorkspaceInfo(null);
      setCharucoCount(0);
      addLog('Calibration reset.', 'system');
    } catch (e) { addLog('Reset failed: ' + e.message, 'error'); }
  }

  async function saveSettings() {
    try {
      await api('settings', 'POST', {
        gripper_marker_id: gripperId,
        wrist_marker_id: wristId,
        aruco_ws_marker_size: wsMarkerSize,
        aruco_robot_marker_size: robotMarkerSize,
        grasp_z: graspZ,
      });
      addLog('Settings saved.', 'success');
    } catch (e) { addLog('Save failed: ' + e.message, 'error'); }
  }

  const active = armConnected || previewing;

  return (
    <>
      <div className="card card-highlight">
        <div className="card-header">
          <h2>🎯 ArUco Calibration</h2>
          <span className={`badge badge-sm ${calBadge.cls}`}>{calBadge.text}</span>
        </div>

        {/* Marker status grid */}
        <div className="marker-grid">
          {ALL_MARKERS.map(mid => (
            <div key={mid} className={`marker-chip ${detectedIds.includes(mid) ? 'detected' : ''}`}>
              <span className="marker-dot" />
              {MARKER_LABELS[mid]}
            </div>
          ))}
        </div>

        {/* Step 1: Intrinsics */}
        <div className="cal-section">
          <h3>1. Camera Intrinsics</h3>
          <p className="hint">
            Use a ChArUco board ({charucoCount} frame{charucoCount !== 1 ? 's' : ''} collected) or estimate from resolution.
          </p>
          <div className="btn-row">
            <button className={`btn btn-sm ${loading === 'collect' ? 'loading' : ''}`}
                    onClick={collectCharuco} disabled={!active || !!loading}>
              Collect Frame
            </button>
            <button className={`btn btn-sm btn-primary ${loading === 'intrinsics' ? 'loading' : ''}`}
                    onClick={calibrateIntrinsics} disabled={!active || charucoCount < 5 || !!loading}>
              Calibrate ({charucoCount}/5+)
            </button>
            <button className={`btn btn-sm btn-ghost ${loading === 'estimate' ? 'loading' : ''}`}
                    onClick={estimateIntrinsics} disabled={!active || !!loading}>
              Estimate
            </button>
          </div>
          {intrinsicsInfo && (
            <span className={`cal-info ${intrinsicsInfo.ok ? 'success' : 'error'}`}>
              {intrinsicsInfo.text}
            </span>
          )}
        </div>

        {/* Step 2: Workspace */}
        <div className="cal-section">
          <h3>2. Workspace Extrinsics</h3>
          <p className="hint">Ensure markers 0-3 are visible on table edges.</p>
          <div className="form-row compact">
            <label>WS Marker Size (m)</label>
            <input type="number" value={wsMarkerSize} step="0.001" min="0.01" max="0.2"
                   onChange={e => setWsMarkerSize(parseFloat(e.target.value) || 0.033)} />
          </div>
          <div className="workspace-dims">
            <div className="form-row compact">
              <label>Table W (m)</label>
              <input type="number" value={tableW} step="0.01" min="0.1" max="2.0"
                     onChange={e => setTableW(parseFloat(e.target.value) || 0.63)} />
            </div>
            <div className="form-row compact">
              <label>Table D (m)</label>
              <input type="number" value={tableD} step="0.01" min="0.1" max="2.0"
                     onChange={e => setTableD(parseFloat(e.target.value) || 0.60)} />
            </div>
          </div>
          <div className="btn-row">
            <button className={`btn btn-sm btn-primary ${loading === 'workspace' ? 'loading' : ''}`}
                    onClick={calibrateWorkspace} disabled={!active || !!loading}>
              Calibrate Workspace
            </button>
          </div>
          {workspaceInfo && (
            <span className={`cal-info ${workspaceInfo.ok ? 'success' : 'error'}`}>
              {workspaceInfo.text}
            </span>
          )}
          {status.reprojection_error != null && !workspaceInfo && (
            <span className="cal-info success">
              Saved RMS: {status.reprojection_error.toFixed(3)}px
            </span>
          )}
        </div>

        {/* Step 3: Tracking mode */}
        <div className="cal-section">
          <h3>3. Tracking Mode</h3>
          <div className="radio-group">
            {[
              ['aruco', 'ArUco Marker'],
              ['blink', 'Sync-Blink'],
              ['marker', 'HSV Color'],
            ].map(([val, label]) => (
              <label key={val} className="radio-label">
                <input type="radio" name="tracking" value={val}
                       checked={trackingMode === val}
                       onChange={() => setTrackingMode(val)} />
                {label}
              </label>
            ))}
          </div>
        </div>

        <div className="btn-row" style={{ marginTop: 12 }}>
          <button className="btn btn-sm btn-danger btn-ghost" onClick={resetCalibration}>
            Reset Calibration
          </button>
        </div>
      </div>

      {/* ArUco Settings (collapsible) */}
      <div className="card">
        <div className="card-header clickable" onClick={() => setSettingsOpen(!settingsOpen)}>
          <h2>⚙️ ArUco Settings</h2>
          <span className="chevron">{settingsOpen ? '▲' : '▼'}</span>
        </div>
        {settingsOpen && (
          <div>
            <div className="form-row compact">
              <label>Gripper Marker ID</label>
              <input type="number" value={gripperId} min="0" max="49"
                     onChange={e => setGripperId(parseInt(e.target.value) || 48)} />
            </div>
            <div className="form-row compact">
              <label>Wrist Marker ID</label>
              <input type="number" value={wristId} min="0" max="49"
                     onChange={e => setWristId(parseInt(e.target.value) || 49)} />
            </div>
            <div className="form-row compact">
              <label>Robot Marker Size (m)</label>
              <input type="number" value={robotMarkerSize} step="0.001" min="0.005" max="0.1"
                     onChange={e => setRobotMarkerSize(parseFloat(e.target.value) || 0.018)} />
            </div>
            <div className="form-row compact">
              <label>Grasp Z Height (m)</label>
              <input type="number" value={graspZ} step="0.001" min="0.001" max="0.1"
                     onChange={e => setGraspZ(parseFloat(e.target.value) || 0.018)} />
            </div>
            <button className="btn btn-sm" onClick={saveSettings}>Save Settings</button>
          </div>
        )}
      </div>
    </>
  );
}
