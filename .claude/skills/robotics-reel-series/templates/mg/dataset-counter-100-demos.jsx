const Component = ({ item }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const props = item.props || {};
  const red = props.red;
  const blue = props.blue;
  const navy = props.navy;
  const revealAt = props.revealAt;

  const sPanel = spring({ frame: frame, fps, config: { damping: 15, stiffness: 150 } });
  const countP = interpolate(frame, [6, 50], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });
  const count = Math.round(countP * props.target);
  const landed = spring({ frame: frame - 50, fps, config: { damping: 8, stiffness: 220 } });
  const numScale = frame < 50 ? 1 : 1 + 0.12 * (1 - landed);
  const sC1 = spring({ frame: frame - revealAt, fps, config: { damping: 12, stiffness: 170 } });
  const sC2 = spring({ frame: frame - revealAt - 8, fps, config: { damping: 12, stiffness: 170 } });
  const sFoot = spring({ frame: frame - revealAt - 16, fps, config: { damping: 16, stiffness: 140 } });
  const exit = interpolate(frame, [durationInFrames - 10, durationInFrames], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.in(Easing.cubic) });
  const scan = (frame * 6) % 760 - 60;

  const rootStyle = { position: "absolute", inset: 0, backgroundColor: "transparent", opacity: 1 - exit, transform: "scale(" + (0.9 + 0.1 * sPanel - 0.05 * exit) + ")" };
  const panel = { position: "absolute", inset: 0, borderRadius: 30, overflow: "hidden", border: "1px solid rgba(110,155,255,0.4)", boxShadow: "0 24px 60px rgba(0,0,0,0.45)", opacity: sPanel };
  const bgStyle = { position: "absolute", inset: 0, backgroundColor: navy, opacity: 0.86 };
  const scanStyle = { position: "absolute", left: 0, right: 0, top: scan, height: 60, backgroundImage: "linear-gradient(180deg, rgba(47,107,255,0) 0%, rgba(47,107,255,0.18) 50%, rgba(47,107,255,0) 100%)" };
  const col = { position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 6 };
  const labelStyle = { display: "flex", alignItems: "center", gap: 10, fontFamily: props.monoFont, fontWeight: 700, fontSize: 28, color: "#9DB7FF", letterSpacing: 6 };
  const recDot = { width: 14, height: 14, borderRadius: 7, backgroundColor: red, boxShadow: "0 0 12px " + red };
  const numStyle = { fontFamily: props.headFont, fontSize: 250, lineHeight: 1.0, color: "#FFFFFF", letterSpacing: 4, textShadow: "-8px 0 0 " + red + ", 8px 0 0 " + blue, transform: "scale(" + numScale + ")" };
  const unitStyle = { fontFamily: props.bodyFont, fontWeight: 700, fontSize: 50, color: "#FFFFFF", letterSpacing: 8, marginTop: -6 };
  const chipRow = { display: "flex", gap: 18, marginTop: 26 };
  const chip1 = { padding: "8px 24px", borderRadius: 10, backgroundColor: red, fontFamily: props.headFont, fontSize: 46, color: "#FFFFFF", letterSpacing: 2, transform: "scale(" + sC1 + ")" };
  const chip2 = { padding: "8px 24px", borderRadius: 10, backgroundColor: blue, fontFamily: props.headFont, fontSize: 46, color: "#FFFFFF", letterSpacing: 2, transform: "scale(" + sC2 + ")" };
  const footStyle = { marginTop: 18, fontFamily: props.monoFont, fontSize: 22, color: "rgba(255,255,255,0.75)", opacity: sFoot, letterSpacing: 1 };
  const corner = { position: "absolute", width: 46, height: 46 };
  const cTL = { ...corner, left: 18, top: 18, borderLeft: "5px solid " + red, borderTop: "5px solid " + red };
  const cTR = { ...corner, right: 18, top: 18, borderRight: "5px solid " + blue, borderTop: "5px solid " + blue };
  const cBL = { ...corner, left: 18, bottom: 18, borderLeft: "5px solid " + blue, borderBottom: "5px solid " + blue };
  const cBR = { ...corner, right: 18, bottom: 18, borderRight: "5px solid " + red, borderBottom: "5px solid " + red };

  return (
    <div style={rootStyle}>
      <div style={panel}>
        <div style={bgStyle} />
        <div style={scanStyle} />
        <div style={cTL} /><div style={cTR} /><div style={cBL} /><div style={cBR} />
        <div style={col}>
          <div style={labelStyle}><div style={recDot} />{props.label}</div>
          <div style={numStyle}>{count}</div>
          <div style={unitStyle}>{props.unit}</div>
          <div style={chipRow}>
            <div style={chip1}>{props.chip1}</div>
            <div style={chip2}>{props.chip2}</div>
          </div>
          <div style={footStyle}>{props.foot}</div>
        </div>
      </div>
    </div>
  );
};
