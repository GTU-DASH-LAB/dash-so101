const Component = ({ item }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const props = item.props || {};
  const red = props.red;
  const blue = props.blue;
  const navy = props.navy;

  const exit = interpolate(frame, [durationInFrames - 6, durationInFrames], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const sHead = spring({ frame: frame, fps, config: { damping: 11, stiffness: 180 } });
  const pulse = 1 + 0.025 * Math.sin(frame / 5);
  const nudge = Math.max(0, Math.sin(frame / 4)) * 8;
  const headScale = Math.max(0, sHead) * pulse;
  const headOpacity = Math.min(1, Math.max(0, sHead * 2));

  const rootStyle = { position: "absolute", inset: 0, backgroundColor: "transparent", opacity: 1 - exit };
  const col = { position: "absolute", left: 0, right: 0, top: 0, bottom: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-start", paddingTop: 12, gap: 22 };
  const head = { display: "flex", alignItems: "center", gap: 14, padding: "4px 28px", borderRadius: 14, backgroundColor: red, fontFamily: props.headFont, fontSize: 76, lineHeight: 1.12, color: "#FFFFFF", letterSpacing: 3, whiteSpace: "nowrap", opacity: headOpacity, transform: "translateY(" + (50 * (1 - sHead)) + "px) scale(" + headScale + ") rotate(-2deg)", boxShadow: "0 16px 36px rgba(0,0,0,0.4)" };
  const arrowWrap = { display: "flex", transform: "translateX(" + nudge + "px)" };
  const grid = { display: "grid", gridTemplateColumns: "430px 430px", gap: 14 };

  const renderChip = (delay, accent, iconName, label, handle) => {
    const s = spring({ frame: frame - delay, fps, config: { damping: 13, stiffness: 170 } });
    const len = Math.max(1, String(handle).length);
    const fs = Math.min(34, Math.floor(306 / (len * 0.6)));
    const chipOpacity = Math.min(1, Math.max(0, s * 1.6));
    const chip = { position: "relative", width: 430, height: 104, borderRadius: 20, overflow: "hidden", display: "flex", alignItems: "center", gap: 16, padding: "0 20px", boxSizing: "border-box", border: "2px solid " + accent, boxShadow: "0 12px 30px rgba(0,0,0,0.38)", opacity: chipOpacity, transform: "translateY(" + (40 * (1 - s)) + "px) scale(" + (0.7 + 0.3 * s) + ")" };
    const bg = { position: "absolute", inset: 0, backgroundColor: navy, opacity: 0.9 };
    const tile = { position: "relative", flexShrink: 0, width: 64, height: 64, borderRadius: 16, backgroundColor: accent, display: "flex", alignItems: "center", justifyContent: "center" };
    const txt = { position: "relative", display: "flex", flexDirection: "column", gap: 2, minWidth: 0 };
    const lab = { fontFamily: props.monoFont, fontWeight: 700, fontSize: 19, letterSpacing: 2.5, color: "#9DB7FF", whiteSpace: "nowrap" };
    const hd = { fontFamily: props.bodyFont, fontWeight: 700, fontSize: fs, lineHeight: 1.1, color: "#FFFFFF", whiteSpace: "nowrap" };
    return (
      <div style={chip}>
        <div style={bg} />
        <div style={tile}><Icon name={iconName} size={36} color="#FFFFFF" strokeWidth={2.2} /></div>
        <div style={txt}><div style={lab}>{label}</div><div style={hd}>{handle}</div></div>
      </div>
    );
  };

  const chipIg = renderChip(6, red, "instagram", props.igLabel, props.igHandle);
  const chipLi = renderChip(10, blue, "linkedin", props.liLabel, props.liHandle);
  const chipGh1 = renderChip(14, red, "github", props.gh1Label, props.gh1);
  const chipGh2 = renderChip(18, blue, "github", props.gh2Label, props.gh2);

  return (
    <div style={rootStyle}>
      <div style={col}>
        <div style={head}>{props.follow}<div style={arrowWrap}><Icon name="arrow-right" size={60} color="#FFFFFF" strokeWidth={3} /></div></div>
        <div style={grid}>
          {chipIg}
          {chipLi}
          {chipGh1}
          {chipGh2}
        </div>
      </div>
    </div>
  );
};
