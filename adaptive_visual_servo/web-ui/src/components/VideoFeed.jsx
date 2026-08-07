import { useRef, useCallback, useEffect, useState } from 'react';

export default function VideoFeed({
  frameData, markers, pickPx, dropPx, clickMode,
  onClickCanvas, onSetPick, onSetDrop, onClear
}) {
  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const [naturalSize, setNaturalSize] = useState({ w: 640, h: 480 });

  const markerCount = Object.keys(markers).length;

  const handleImgLoad = useCallback(() => {
    const img = imgRef.current;
    if (img && img.naturalWidth > 0) {
      setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
    }
  }, []);

  // Draw pick/drop overlays
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const sx = rect.width / naturalSize.w;
    const sy = rect.height / naturalSize.h;

    const drawCross = (px, color) => {
      const x = px[0] * sx, y = px[1] * sy;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x - 12, y); ctx.lineTo(x + 12, y);
      ctx.moveTo(x, y - 12); ctx.lineTo(x, y + 12);
      ctx.stroke();
      // circle
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.stroke();
    };

    if (pickPx) drawCross(pickPx, '#ef4444');
    if (dropPx) drawCross(dropPx, '#a855f7');
  }, [pickPx, dropPx, naturalSize, frameData]);

  const handleCanvasClick = useCallback((e) => {
    if (!clickMode) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const sx = naturalSize.w / rect.width;
    const sy = naturalSize.h / rect.height;
    const px = [(e.clientX - rect.left) * sx, (e.clientY - rect.top) * sy];
    onClickCanvas(clickMode, px);
  }, [clickMode, naturalSize, onClickCanvas]);

  const showCanvas = clickMode || pickPx || dropPx;

  return (
    <div className="card">
      <div className="card-header">
        <h2>📹 Live Feed</h2>
        <span className="marker-count">{markerCount} marker{markerCount !== 1 ? 's' : ''}</span>
      </div>
      <div className="video-container">
        <img
          ref={imgRef}
          className={`video-feed ${frameData ? 'active' : ''}`}
          src={frameData ? `data:image/jpeg;base64,${frameData}` : undefined}
          alt="Camera Feed"
          onLoad={handleImgLoad}
        />
        <div className={`video-overlay ${frameData ? 'hidden' : ''}`}>
          <span>No video — connect arm or start preview</span>
        </div>
        <canvas
          ref={canvasRef}
          className={`click-canvas ${showCanvas ? 'active' : ''}`}
          style={{ pointerEvents: clickMode ? 'auto' : 'none' }}
          onClick={handleCanvasClick}
        />
      </div>
      <div className="video-toolbar">
        <button className="btn btn-sm" onClick={onSetPick}>🎯 Set Pick</button>
        <button className="btn btn-sm" onClick={onSetDrop}>📍 Set Drop</button>
        <button className="btn btn-sm btn-ghost" onClick={onClear}>Clear</button>
        <div className="spacer" />
        <span className="point-info">
          pick: {pickPx ? `[${pickPx[0].toFixed(0)},${pickPx[1].toFixed(0)}]` : '—'}
          {' | '}
          drop: {dropPx ? `[${dropPx[0].toFixed(0)},${dropPx[1].toFixed(0)}]` : '—'}
        </span>
      </div>
    </div>
  );
}
