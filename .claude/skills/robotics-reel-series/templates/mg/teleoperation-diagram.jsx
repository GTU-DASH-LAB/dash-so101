const Component = ({ item }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const props = item.props || {};
  const red = props.red;
  const blue = props.blue;
  const navy = props.navy;

  const sTitle = spring({ frame: frame, fps, config: { damping: 13, stiffness: 170 } });
  const sCard = spring({ frame: frame - 6, fps, config: { damping: 16, stiffness: 140 } });
  const sL = spring({ frame: frame - 14, fps, config: { damping: 12, stiffness: 160 } });
  const sF = spring({ frame: frame - 30, fps, config: { damping: 12, stiffness: 160 } });
  const lineGrow = interpolate(frame, [20, 40], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });
  const exit = interpolate(frame, [durationInFrames - 10, durationInFrames], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.in(Easing.cubic) });
  const t1 = ((frame - 40) % 24 + 24) % 24 / 24;
  const t2 = ((frame - 52) % 24 + 24) % 24 / 24;
  const pulsesOn = frame > 40 ? 1 : 0;
  const glowF = 0.5 + 0.5 * Math.sin(frame / 5);

  const rootStyle = { position: "absolute", inset: 0, backgroundColor: "transparent", opacity: 1 - exit, transform: "translateY(" + (30 * exit) + "px)" };
  const titleStyle = {
    position: "absolute", left: 0, top: 0, padding: "2px 28px", borderRadius: 10, backgroundColor: blue,
    fontFamily: props.headFont, fontSize: 88, lineHeight: 1.15, color: "#FFFFFF", letterSpacing: 4,
    transform: "translateX(" + (-700 * (1 - sTitle)) + "px) rotate(-2deg)", boxShadow: "0 12px 26px rgba(0,0,0,0.3)",
  };
  const cardStyle = {
    position: "absolute", left: 0, right: 0, top: 128, bottom: 0, borderRadius: 26, overflow: "hidden",
    border: "1px solid rgba(110,155,255,0.35)", boxShadow: "0 16px 40px rgba(0,0,0,0.35)",
    opacity: sCard, transform: "translateY(" + (30 * (1 - sCard)) + "px)",
  };
  const bgStyle = { position: "absolute", inset: 0, backgroundColor: navy, opacity: 0.86 };
  const rowStyle = { position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 30px" };
  const nodeL = { display: "flex", flexDirection: "column", alignItems: "center", gap: 8, width: 300, transform: "scale(" + sL + ")" };
  const nodeF = { display: "flex", flexDirection: "column", alignItems: "center", gap: 8, width: 300, transform: "scale(" + sF + ")" };
  const ringL = { width: 96, height: 96, borderRadius: 24, display: "flex", alignItems: "center", justifyContent: "center", backgroundColor: red, boxShadow: "0 0 " + (14 + 16 * glowF) + "px rgba(255,45,85,0.6)" };
  const ringF = { width: 96, height: 96, borderRadius: 24, display: "flex", alignItems: "center", justifyContent: "center", backgroundColor: blue, boxShadow: "0 0 " + (14 + 16 * (1 - glowF)) + "px rgba(47,107,255,0.7)" };
  const labelStyle = { fontFamily: props.bodyFont, fontWeight: 700, fontSize: 32, color: "#FFFFFF", letterSpacing: 1 };
  const subStyle = { fontFamily: props.monoFont, fontWeight: 500, fontSize: 22, color: "#9DB7FF" };
  const linkWrap = { position: "relative", width: 230, height: 40 };
  const linkLine = { position: "absolute", left: 0, top: 18, height: 4, width: (230 * lineGrow) + "px", borderRadius: 2, backgroundImage: "linear-gradient(90deg, " + red + ", " + blue + ")" };
  const p1 = { position: "absolute", top: 12, left: (t1 * 214) + "px", width: 16, height: 16, borderRadius: 8, backgroundColor: "#FFFFFF", opacity: pulsesOn, boxShadow: "0 0 14px #FFFFFF" };
  const p2 = { position: "absolute", top: 12, left: (t2 * 214) + "px", width: 16, height: 16, borderRadius: 8, backgroundColor: "#FFFFFF", opacity: pulsesOn * 0.7, boxShadow: "0 0 14px #FFFFFF" };
  const sigStyle = { position: "absolute", left: 0, right: 0, top: -12, textAlign: "center", fontFamily: props.monoFont, fontSize: 18, color: "#9DB7FF", letterSpacing: 2, opacity: lineGrow };

  return (
    <div style={rootStyle}>
      <div style={cardStyle}>
        <div style={bgStyle} />
        <div style={rowStyle}>
          <div style={nodeL}>
            <div style={ringL}><Icon name="hand" size={54} color="#FFFFFF" strokeWidth={2.2} /></div>
            <div style={labelStyle}>{props.leader}</div>
            <div style={subStyle}>{props.leaderSub}</div>
          </div>
          <div style={linkWrap}>
            <div style={sigStyle}>30 Hz</div>
            <div style={linkLine} />
            <div style={p1} />
            <div style={p2} />
          </div>
          <div style={nodeF}>
            <div style={ringF}><Icon name="bot" size={54} color="#FFFFFF" strokeWidth={2.2} /></div>
            <div style={labelStyle}>{props.follower}</div>
            <div style={subStyle}>{props.followerSub}</div>
          </div>
        </div>
      </div>
      <div style={titleStyle}>{props.title}</div>
    </div>
  );
};
