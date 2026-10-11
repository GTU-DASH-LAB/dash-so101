const Component = ({ item }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const props = item.props || {};
  const red = props.red;
  const blue = props.blue;
  const navy = props.navy;

  const s = spring({ frame: frame, fps, config: { damping: 15, stiffness: 170 } });
  const exit = interpolate(frame, [durationInFrames - 8, durationInFrames], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const blink = Math.floor(frame / 12) % 2 === 0 ? 1 : 0.25;

  const rootStyle = { position: "absolute", inset: 0, backgroundColor: "transparent", opacity: s * (1 - exit) };
  const tab = { position: "absolute", left: 8, top: 0, height: 52, display: "flex", alignItems: "center", gap: 12, padding: "0 18px", borderRadius: 12, overflow: "hidden" };
  const tabBg = { position: "absolute", inset: 0, backgroundColor: navy, opacity: 0.9 };
  const dot = { position: "relative", width: 14, height: 14, borderRadius: 7, backgroundColor: red, opacity: blink, boxShadow: "0 0 10px " + red };
  const tabText = { position: "relative", fontFamily: props.monoFont, fontWeight: 700, fontSize: 26, color: "#FFFFFF", letterSpacing: 2 };
  const border = { position: "absolute", left: 4, top: 59, width: 452, height: 341, borderRadius: 26, border: "4px solid " + blue, boxShadow: "0 0 24px rgba(47,107,255,0.55)" };
  const corner = { position: "absolute", width: 40, height: 40 };
  const cTL = { ...corner, left: -2, top: 53, borderLeft: "6px solid " + red, borderTop: "6px solid " + red, borderTopLeftRadius: 28 };
  const cBR = { ...corner, right: -2, bottom: -2, borderRight: "6px solid " + red, borderBottom: "6px solid " + red, borderBottomRightRadius: 28 };

  return (
    <div style={rootStyle}>
      <div style={border} />
      <div style={cTL} />
      <div style={cBR} />
      <div style={tab}><div style={tabBg} /><div style={dot} /><div style={tabText}>{props.label}</div></div>
    </div>
  );
};
