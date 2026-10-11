const Component = ({ item }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const props = item.props || {};
  const red = props.red;
  const blue = props.blue;
  const navy = props.navy;

  const sCard = spring({ frame: frame, fps, config: { damping: 18, stiffness: 140 } });
  const sName = spring({ frame: frame - 6, fps, config: { damping: 16, stiffness: 150 } });
  const sRole = spring({ frame: frame - 11, fps, config: { damping: 16, stiffness: 150 } });
  const sOrg = spring({ frame: frame - 15, fps, config: { damping: 16, stiffness: 150 } });
  const exit = interpolate(frame, [durationInFrames - 10, durationInFrames], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.in(Easing.cubic) });
  const reveal = Math.max(0, Math.min(1, sCard)) * (1 - exit);
  const clipRight = (1 - reveal) * 100;
  const shine = interpolate(frame, [10, 40], [-30, 130], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  const rootStyle = { position: "absolute", inset: 0, backgroundColor: "transparent" };
  const cardStyle = {
    position: "absolute", left: 0, top: 0, right: 0, bottom: 0,
    display: "flex", flexDirection: "row", alignItems: "stretch",
    borderRadius: 24, overflow: "hidden",
    border: "1px solid rgba(110,155,255,0.35)",
    boxShadow: "0 16px 40px rgba(0,0,0,0.35)",
    clipPath: "inset(0 " + clipRight + "% 0 0 round 24px)",
  };
  const bgStyle = { position: "absolute", inset: 0, backgroundColor: navy, opacity: 0.88 };
  const barStyle = { position: "relative", width: 14, backgroundImage: "linear-gradient(180deg, " + red + " 0%, " + red + " 50%, " + blue + " 50%, " + blue + " 100%)" };
  const textCol = { position: "relative", flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", gap: 8, padding: "22px 34px" };
  const nameStyle = {
    fontFamily: props.nameFont, fontWeight: 700, fontSize: 58, lineHeight: 1.05, color: "#FFFFFF", letterSpacing: 1.5,
    opacity: sName, transform: "translateY(" + (20 * (1 - sName)) + "px)",
  };
  const roleStyle = {
    fontFamily: props.monoFont, fontWeight: 700, fontSize: 27, color: "#9DB7FF", letterSpacing: 0.5,
    opacity: sRole, transform: "translateY(" + (16 * (1 - sRole)) + "px)", whiteSpace: "normal", overflowWrap: "break-word",
  };
  const orgRow = { display: "flex", alignItems: "center", gap: 12, opacity: sOrg, transform: "translateY(" + (16 * (1 - sOrg)) + "px)" };
  const orgDot = { width: 12, height: 12, borderRadius: 6, backgroundColor: red, boxShadow: "0 0 10px " + red };
  const orgStyle = { fontFamily: props.nameFont, fontWeight: 600, fontSize: 30, color: "#FFFFFF" };
  const shineStyle = {
    position: "absolute", top: 0, bottom: 0, width: 120, left: shine + "%",
    backgroundImage: "linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.12) 50%, rgba(255,255,255,0) 100%)",
  };

  return (
    <div style={rootStyle}>
      <div style={cardStyle}>
        <div style={bgStyle} />
        <div style={barStyle} />
        <div style={textCol}>
          <div style={nameStyle}>{props.name}</div>
          <div style={roleStyle}>{props.role}</div>
          <div style={orgRow}><div style={orgDot} /><div style={orgStyle}>{props.org}</div></div>
        </div>
        <div style={shineStyle} />
      </div>
    </div>
  );
};
