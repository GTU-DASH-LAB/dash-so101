export default function SpeedTuning({ speed, setSpeed, probes, setProbes }) {
  return (
    <div className="card">
      <div className="card-header"><h2>⚡ Speed & Tuning</h2></div>
      <div className="slider-group">
        <label>Arm Speed</label>
        <input type="range" min="1" max="4" step="0.5" value={speed}
               onChange={e => setSpeed(parseFloat(e.target.value))} />
        <span className="slider-value">{speed.toFixed(1)}×</span>
      </div>
      <div className="slider-group">
        <label>Babble Probes</label>
        <input type="range" min="10" max="30" step="1" value={probes}
               onChange={e => setProbes(parseInt(e.target.value))} />
        <span className="slider-value">{probes}</span>
      </div>
    </div>
  );
}
