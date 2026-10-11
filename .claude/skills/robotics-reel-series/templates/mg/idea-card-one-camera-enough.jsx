const Component = ({ item }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const props = item.props || {};
  const red = props.red;
  const blue = props.blue;
  const navy = props.navy;

  const sChip = spring({ frame: frame, fps, config: { damping: 12, stiffness: 180 } });
  const sCard = spring({ frame: frame - 5, fps, config: { damping: 16, stiffness: 140 } });
  const sL1 = spring({ frame: frame - 40, fps, config: { damping: 16, stiffness: 150 } });
  const sMain = spring({ frame: frame - 60, fps, config: { damping: 13, stiffness: 160 } });
  const sRest = spring({ frame: frame - 70, fps, config: { damping: 13, stiffness: 160 } });
  const hlPulse = spring({ frame: frame - props.hlAt, fps, config: { damping: 7, stiffness: 220 } });
  const hlScale = frame < props.hlAt ? 1 : 1 + 0.12 * (1 - hlPulse);
  const exit = interpolate(frame, [durationInFrames - 10, durationInFrames], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.in(Easing.cubic) });
  const bulb = 0.6 + 0.4 * Math.sin(frame / 5);

  const rootStyle = { position: "absolute", inset: 0, backgroundColor: "transparent", opacity: 1 - exit, transform: "translateY(" + (40 * exit) + "px)" };
  const chip = { position: "absolute", left: 0, top: 0, display: "flex", alignItems: "center", gap: 10, padding: "4px 24px 4px 16px", borderRadius: 10, backgroundColor: blue, fontFamily: props.headFont, fontSize: 56, lineHeight: 1.15, color: "#FFFFFF", letterSpacing: 3, transform: "translateX(" + (-500 * (1 - sChip)) + "px) rotate(-2deg)", boxShadow: "0 12px 26px rgba(0,0,0,0.3)", whiteSpace: "nowrap" };
  const bulbWrap = { opacity: bulb };
  const card = { position: "absolute", left: 0, right: 0, top: 84, bottom: 0, borderRadius: 26, overflow: "hidden", border: "1px solid rgba(110,155,255,0.35)", boxShadow: "0 16px 40px rgba(0,0,0,0.35)", opacity: sCard, transform: "translateY(" + (30 * (1 - sCard)) + "px)" };
  const bg = { position: "absolute", inset: 0, backgroundColor: navy, opacity: 0.88 };
  const col = { position: "absolute", left: 0, right: 0, top: 0, bottom: 0, display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "flex-start", gap: 12, padding: "18px 34px" };
  const l1 = { fontFamily: props.bodyFont, fontWeight: 600, fontSize: 32, lineHeight: 1.15, color: "#C9D3FF", opacity: sL1, transform: "translateY(" + (14 * (1 - sL1)) + "px)" };
  const mainRow = { display: "flex", alignItems: "center", gap: 16, flexWrap: "nowrap" };
  const hl = { fontFamily: props.headFont, fontSize: 60, lineHeight: 1.12, color: "#FFFFFF", letterSpacing: 2, padding: "0 14px", borderRadius: 8, backgroundColor: red, opacity: sMain, transform: "translateY(" + (16 * (1 - sMain)) + "px) scale(" + hlScale + ")", transformOrigin: "left center", whiteSpace: "nowrap" };
  const rest = { fontFamily: props.headFont, fontSize: 60, lineHeight: 1.12, color: "#FFFFFF", letterSpacing: 2, opacity: sRest, transform: "translateY(" + (16 * (1 - sRest)) + "px)", whiteSpace: "nowrap" };

  return (
    <div style={rootStyle}>
      <div style={card}>
        <div style={bg} />
        <div style={col}>
          <div style={l1}>{props.line1}</div>
          <div style={mainRow}><div style={hl}>{props.hl}</div><div style={rest}>{props.rest}</div></div>
        </div>
      </div>
      <div style={chip}><div style={bulbWrap}><Icon name="lightbulb" size={46} color="#FFFFFF" strokeWidth={2.6} /></div>{props.chip}</div>
    </div>
  );
};
