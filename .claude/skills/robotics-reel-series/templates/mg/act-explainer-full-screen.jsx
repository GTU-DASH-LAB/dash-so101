const Component = ({ item }) => {
  const LOSS = [1.099, -0.397, -0.585, -0.693, -0.77, -0.83, -0.866, -0.879, -0.907, -0.924, -0.914, -0.924, -0.932, -0.921, -0.903, -0.917, -0.924, -0.91, -0.928, -0.955, -0.893, -0.936, -0.932, -0.991, -0.967, -0.971, -0.979, -1.004, -1.013, -0.963, -1.009, -1.027, -0.987, -0.991, -1.056, -0.996, -1.004, -1.071, -1.009, -1.046];
  const BARS = 50;
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const props = item.props || {};
  const red = props.red;
  const blue = props.blue;
  const navy = props.navy;
  const canvasRef = React.useRef(null);

  const enter = interpolate(frame, [0, 8], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const exit = interpolate(frame, [durationInFrames - 8, durationInFrames], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const sTitle = spring({ frame: frame - 2, fps, config: { damping: 12, stiffness: 170 } });
  const sSub = spring({ frame: frame - 8, fps, config: { damping: 16, stiffness: 150 } });
  const sIn = spring({ frame: frame - 14, fps, config: { damping: 15, stiffness: 150 } });
  const sA1 = interpolate(frame, [26, 36], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const sCore = spring({ frame: frame - 32, fps, config: { damping: 13, stiffness: 160 } });
  const sA2 = interpolate(frame, [48, 58], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const sOut = spring({ frame: frame - 52, fps, config: { damping: 16, stiffness: 150 } });
  const fill = interpolate(frame, [58, 100], [0, BARS], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const sTrain = spring({ frame: frame - 70, fps, config: { damping: 16, stiffness: 140 } });
  const draw = interpolate(frame, [80, 140], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.cubic) });
  const coreGlow = 0.5 + 0.5 * Math.sin(frame / 4);
  const arrowBob = Math.sin(frame / 5) * 5;

  React.useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const W = c.width; const H = c.height;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(157,183,255,0.18)"; ctx.lineWidth = 1;
    for (let g = 1; g < 4; g++) { ctx.beginPath(); ctx.moveTo(0, (H * g) / 4); ctx.lineTo(W, (H * g) / 4); ctx.stroke(); }
    const mn = -1.12; const mx = 1.15;
    const n = LOSS.length;
    const upto = Math.max(1, draw * (n - 1));
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, red); grad.addColorStop(1, blue);
    ctx.strokeStyle = grad; ctx.lineWidth = 5; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.beginPath();
    let lx = 0; let ly = 0;
    for (let i = 0; i <= Math.floor(upto); i++) {
      const x = 8 + (i / (n - 1)) * (W - 16);
      const y = 8 + (1 - (LOSS[i] - mn) / (mx - mn)) * (H - 16);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      lx = x; ly = y;
    }
    ctx.stroke();
    if (draw > 0) { ctx.fillStyle = "#FFFFFF"; ctx.beginPath(); ctx.arc(lx, ly, 8, 0, Math.PI * 2); ctx.fill(); }
  }, [frame, red, blue, draw]);

  const bgOpacity = props.transparentBackground ? 0 : 1;
  const rootStyle = { position: "absolute", inset: 0, overflow: "hidden", opacity: enter * (1 - exit), backgroundColor: "transparent" };
  const bg = { position: "absolute", inset: 0, backgroundColor: navy, opacity: bgOpacity };
  const grid = { position: "absolute", inset: 0, opacity: bgOpacity, backgroundImage: "linear-gradient(rgba(47,107,255,0.09) 2px, transparent 2px), linear-gradient(90deg, rgba(47,107,255,0.09) 2px, transparent 2px)", backgroundSize: "60px 60px" };
  const glowBlue = { position: "absolute", left: -300, top: -260, width: 900, height: 900, borderRadius: 450, opacity: 0.55 * bgOpacity, backgroundImage: "radial-gradient(circle, rgba(47,107,255,0.55) 0%, rgba(47,107,255,0) 70%)" };
  const glowRed = { position: "absolute", right: -320, bottom: -200, width: 900, height: 900, borderRadius: 450, opacity: 0.45 * bgOpacity, backgroundImage: "radial-gradient(circle, rgba(255,45,85,0.5) 0%, rgba(255,45,85,0) 70%)" };

  const titleStyle = { position: "absolute", left: 96, top: 110, fontFamily: props.headFont, fontSize: 180, lineHeight: 1, color: "#FFFFFF", letterSpacing: 6, textShadow: "-8px 0 0 " + red + ", 8px 0 0 " + blue, transform: "translateY(" + (-40 * (1 - sTitle)) + "px)", opacity: sTitle };
  const subStyle = { position: "absolute", left: 100, top: 302, fontFamily: props.bodyFont, fontWeight: 700, fontSize: 40, color: "#9DB7FF", letterSpacing: 5, opacity: sSub, transform: "translateX(" + (-40 * (1 - sSub)) + "px)" };

  const inLabel = { position: "absolute", left: 100, top: 382, fontFamily: props.monoFont, fontWeight: 700, fontSize: 24, color: red, letterSpacing: 4, opacity: sIn };
  const videoFrame = { position: "absolute", left: 92, top: 412, width: 376, height: 286, borderRadius: 22, border: "4px solid " + blue, boxShadow: "0 0 30px rgba(47,107,255,0.5)", opacity: sIn, transform: "scale(" + (0.85 + 0.15 * sIn) + ")" };
  const inText = { position: "absolute", left: 512, top: 440, display: "flex", flexDirection: "column", gap: 10, opacity: sIn, transform: "translateX(" + (40 * (1 - sIn)) + "px)" };
  const inL1 = { fontFamily: props.bodyFont, fontWeight: 700, fontSize: 46, color: "#FFFFFF", lineHeight: 1.05 };
  const inL2 = { fontFamily: props.bodyFont, fontWeight: 600, fontSize: 36, color: "#FFFFFF", opacity: 0.9 };
  const inL3 = { fontFamily: props.monoFont, fontSize: 24, color: "#9DB7FF" };

  const arrow1 = { position: "absolute", left: 510, top: 704 + arrowBob, opacity: sA1 };
  const coreStyle = { position: "absolute", left: 92, top: 752, width: 896, height: 108, borderRadius: 22, border: "3px solid " + blue, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 34px", boxShadow: "0 0 " + (20 + 26 * coreGlow) + "px rgba(47,107,255,0.55)", transform: "scale(" + sCore + ")" };
  const coreBg = { position: "absolute", inset: 0, borderRadius: 20, backgroundImage: "linear-gradient(90deg, rgba(47,107,255,0.35), rgba(255,45,85,0.25))" };
  const coreText = { position: "relative", fontFamily: props.headFont, fontSize: 66, color: "#FFFFFF", letterSpacing: 4 };
  const coreSub = { position: "relative", fontFamily: props.monoFont, fontSize: 22, color: "#FFFFFF", opacity: 0.85, textAlign: "right" };
  const arrow2 = { position: "absolute", left: 510, top: 862 + arrowBob, opacity: sA2 };

  const outLabel = { position: "absolute", left: 100, top: 912, fontFamily: props.monoFont, fontWeight: 700, fontSize: 24, color: blue, letterSpacing: 3, opacity: sOut };
  const barsWrap = { position: "absolute", left: 100, top: 952, width: 880, height: 120, display: "flex", alignItems: "flex-end", gap: 5, opacity: sOut };
  const bars = [];
  for (let i = 0; i < BARS; i++) {
    const on = Math.max(0, Math.min(1, fill - i));
    const h = 30 + 80 * (0.5 + 0.5 * Math.sin(i / 6 + 0.6));
    const col = interpolateColors(i / (BARS - 1), [0, 1], [red, blue]);
    const st = { width: 12.6, height: h, borderRadius: 4, backgroundColor: col, opacity: 0.15 + 0.85 * on, transform: "scaleY(" + (0.3 + 0.7 * on) + ")", transformOrigin: "bottom" };
    bars.push(<div key={i} style={st} />);
  }

  const trainCard = { position: "absolute", left: 92, top: 1336, width: 896, height: 270, borderRadius: 26, border: "1px solid rgba(110,155,255,0.35)", overflow: "hidden", opacity: sTrain, transform: "translateY(" + (40 * (1 - sTrain)) + "px)" };
  const trainBg = { position: "absolute", inset: 0, backgroundColor: "#0B1236", opacity: 0.95 };
  const trainHead = { position: "absolute", left: 28, top: 20, right: 28, display: "flex", alignItems: "center", justifyContent: "space-between" };
  const trainLabel = { fontFamily: props.monoFont, fontWeight: 700, fontSize: 24, color: "#9DB7FF", letterSpacing: 2 };
  const chipRow = { display: "flex", gap: 10 };
  const chipR = { padding: "4px 14px", borderRadius: 8, backgroundColor: red, fontFamily: props.headFont, fontSize: 28, color: "#FFFFFF", letterSpacing: 1 };
  const chipB = { padding: "4px 14px", borderRadius: 8, backgroundColor: blue, fontFamily: props.headFont, fontSize: 28, color: "#FFFFFF", letterSpacing: 1 };
  const canvasStyle = { position: "absolute", left: 28, top: 78, width: 840, height: 150 };
  const lossLabel = { position: "absolute", right: 30, bottom: 16, fontFamily: props.monoFont, fontWeight: 700, fontSize: 22, color: "#FFFFFF", opacity: draw };

  return (
    <div style={rootStyle}>
      <div style={bg} />
      <div style={grid} />
      <div style={glowBlue} />
      <div style={glowRed} />
      <div style={titleStyle}>{props.title}</div>
      <div style={subStyle}>{props.subtitle}</div>
      <div style={inLabel}>INPUT</div>
      <div style={videoFrame} />
      <div style={inText}>
        <div style={inL1}>{props.inLine1}</div>
        <div style={inL2}>{props.inLine2}</div>
        <div style={inL3}>{props.inLine3}</div>
      </div>
      <div style={arrow1}><Icon name="chevrons-down" size={48} color="#FFFFFF" strokeWidth={3} /></div>
      <div style={coreStyle}>
        <div style={coreBg} />
        <div style={coreText}>{props.core}</div>
        <div style={coreSub}>{props.coreSub}</div>
      </div>
      <div style={arrow2}><Icon name="chevrons-down" size={48} color="#FFFFFF" strokeWidth={3} /></div>
      <div style={outLabel}>{props.outLabel}</div>
      <div style={barsWrap}>{bars}</div>
      <div style={trainCard}>
        <div style={trainBg} />
        <div style={trainHead}>
          <div style={trainLabel}>{props.trainLabel}</div>
          <div style={chipRow}><div style={chipR}>{props.stat1}</div><div style={chipB}>{props.stat2}</div></div>
        </div>
        <canvas ref={canvasRef} width={840} height={150} style={canvasStyle} />
        <div style={lossLabel}>{props.lossLabel}</div>
      </div>
    </div>
  );
};
