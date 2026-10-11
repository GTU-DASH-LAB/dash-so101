const Component = ({ item }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const props = item.props || {};
  const red = props.red;
  const blue = props.blue;
  const navy = props.navy;

  const sChip = spring({ frame: frame, fps, config: { damping: 11, stiffness: 190 } });
  const sSub = spring({ frame: frame - 8, fps, config: { damping: 16, stiffness: 150 } });
  const exit = interpolate(frame, [durationInFrames - 8, durationInFrames], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.in(Easing.cubic) });
  const ping = (frame % 30) / 30;

  const rootStyle = { position: "absolute", inset: 0, backgroundColor: "transparent", opacity: 1 - exit, transform: "translateX(" + (-60 * exit) + "px)" };
  const col = { position: "absolute", left: 0, top: 6, right: 0, display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 14 };
  const chip = { display: "flex", alignItems: "center", gap: 16, padding: "8px 26px 8px 12px", borderRadius: 14, backgroundColor: red, boxShadow: "0 14px 30px rgba(0,0,0,0.35)", transform: "scale(" + sChip + ") rotate(-2deg)", transformOrigin: "left center", whiteSpace: "nowrap" };
  const iconBox = { position: "relative", width: 78, height: 78, borderRadius: 39, backgroundColor: navy, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 };
  const pingStyle = { position: "absolute", left: 0, top: 0, width: 78, height: 78, borderRadius: 39, border: "3px solid #FFFFFF", opacity: 1 - ping, transform: "scale(" + (1 + 0.5 * ping) + ")" };
  const chipText = { fontFamily: props.headFont, fontSize: 74, lineHeight: 1.1, color: "#FFFFFF", letterSpacing: 3, whiteSpace: "nowrap" };
  const subWrap = { position: "relative", display: "flex", alignItems: "center", gap: 12, padding: "10px 20px", borderRadius: 10, overflow: "hidden", opacity: sSub, transform: "translateY(" + (16 * (1 - sSub)) + "px)" };
  const subBg = { position: "absolute", inset: 0, backgroundColor: navy, opacity: 0.86 };
  const subBar = { position: "relative", width: 8, height: 30, borderRadius: 4, backgroundColor: blue };
  const subText = { position: "relative", fontFamily: props.monoFont, fontWeight: 700, fontSize: 30, color: "#FFFFFF", whiteSpace: "nowrap" };

  return (
    <div style={rootStyle}>
      <div style={col}>
        <div style={chip}>
          <div style={iconBox}><div style={pingStyle} /><Icon name="camera" size={46} color="#FFFFFF" strokeWidth={2.4} /></div>
          <div style={chipText}>{props.chip}</div>
        </div>
        <div style={subWrap}><div style={subBg} /><div style={subBar} /><div style={subText}>{props.sub}</div></div>
      </div>
    </div>
  );
};
