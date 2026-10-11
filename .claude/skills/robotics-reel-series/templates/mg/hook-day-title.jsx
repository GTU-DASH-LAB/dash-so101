const Component = ({ item }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const props = item.props || {};
  const red = props.red;
  const blue = props.blue;
  const navy = props.navy;

  const sLabel = spring({ frame: frame, fps, config: { damping: 16, stiffness: 170 } });
  const s1 = spring({ frame: frame - 4, fps, config: { damping: 12, stiffness: 150 } });
  const s2 = spring({ frame: frame - 12, fps, config: { damping: 14, stiffness: 160 } });
  const s3 = spring({ frame: frame - 19, fps, config: { damping: 14, stiffness: 160 } });
  const exit = interpolate(frame, [durationInFrames - 10, durationInFrames], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.in(Easing.cubic) });
  const dotOn = Math.floor(frame / 12) % 2 === 0 ? 1 : 0.2;
  const glitch = frame > 4 && frame < 12 ? (frame % 2 === 0 ? 8 : -8) : 0;
  const pulse = 1 + 0.015 * Math.sin(frame / 6);

  const rootStyle = { position: "absolute", inset: 0, backgroundColor: "transparent", overflow: "hidden" };
  const stackStyle = {
    position: "absolute", left: 0, top: 0, width: "100%", height: "100%",
    display: "flex", flexDirection: "column", alignItems: "flex-start", justifyContent: "flex-start", gap: 9,
    transform: "rotate(-2deg) translateY(" + (-60 * exit) + "px)", transformOrigin: "left top", opacity: 1 - exit,
    paddingTop: 6, paddingLeft: 10,
  };
  const labelStyle = {
    display: "flex", alignItems: "center", gap: 12, padding: "6px 16px", borderRadius: 10,
    backgroundColor: "rgba(7,11,36,0.82)", border: "1px solid rgba(255,255,255,0.18)",
    fontFamily: props.monoFont, fontSize: 24, fontWeight: 700, color: "#FFFFFF", letterSpacing: 3, whiteSpace: "nowrap",
    opacity: sLabel, transform: "translateY(" + (-30 * (1 - sLabel)) + "px)",
  };
  const dotStyle = { width: 16, height: 16, borderRadius: 8, backgroundColor: red, opacity: dotOn, boxShadow: "0 0 14px " + red };
  const bigChip = {
    display: "flex", alignItems: "center", padding: "0 36px 0 26px", backgroundColor: navy,
    borderLeft: "12px solid " + red, borderRadius: 12,
    transform: "translateX(" + (-1100 * (1 - s1) + glitch) + "px) scale(" + pulse + ")", transformOrigin: "left center",
    boxShadow: "0 16px 36px rgba(0,0,0,0.35)",
  };
  const bigText = {
    fontFamily: props.headFont, fontSize: 168, lineHeight: 1.04, color: "#FFFFFF", letterSpacing: 4, whiteSpace: "nowrap",
    textShadow: "-6px 0 0 " + red + ", 6px 0 0 " + blue,
  };
  const chipRow = { display: "flex", flexDirection: "row", alignItems: "center", gap: 12 };
  const chip2 = {
    padding: "4px 26px", backgroundColor: red, borderRadius: 10, whiteSpace: "nowrap",
    fontFamily: props.headFont, fontSize: 52, lineHeight: 1.1, color: "#FFFFFF", letterSpacing: 2,
    transform: "translateX(" + (-1100 * (1 - s2)) + "px)", boxShadow: "0 12px 26px rgba(0,0,0,0.3)",
  };
  const chip3 = {
    padding: "4px 26px", backgroundColor: blue, borderRadius: 10, whiteSpace: "nowrap",
    fontFamily: props.headFont, fontSize: 52, lineHeight: 1.1, color: "#FFFFFF", letterSpacing: 2,
    transform: "translateX(" + (1100 * (1 - s3)) + "px)", boxShadow: "0 12px 26px rgba(0,0,0,0.3)",
  };

  return (
    <div style={rootStyle}>
      <div style={stackStyle}>
        <div style={labelStyle}><div style={dotStyle} />{props.label}</div>
        <div style={bigChip}><div style={bigText}>{props.title}</div></div>
        <div style={chipRow}>
          <div style={chip2}>{props.line1}</div>
          <div style={chip3}>{props.line2}</div>
        </div>
      </div>
    </div>
  );
};
