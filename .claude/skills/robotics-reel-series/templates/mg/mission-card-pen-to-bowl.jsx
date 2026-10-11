const Component = ({ item }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const props = item.props || {};
  const red = props.red;
  const blue = props.blue;
  const navy = props.navy;

  const sChip = spring({ frame: frame, fps, config: { damping: 14, stiffness: 170 } });
  const sCard = spring({ frame: frame - 5, fps, config: { damping: 16, stiffness: 140 } });
  const sA = spring({ frame: frame - 12, fps, config: { damping: 12, stiffness: 160 } });
  const sB = spring({ frame: frame - 40, fps, config: { damping: 12, stiffness: 160 } });
  const arrowGrow = interpolate(frame, [22, 42], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });
  const exit = interpolate(frame, [durationInFrames - 10, durationInFrames], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.in(Easing.cubic) });
  const travel = (frame % 30) / 30;
  const bob = Math.sin(frame / 7) * 4;

  const rootStyle = { position: "absolute", inset: 0, backgroundColor: "transparent", opacity: 1 - exit, transform: "translateY(" + (-40 * exit) + "px)" };
  const chipStyle = {
    position: "absolute", left: 0, top: 0, padding: "4px 26px", borderRadius: 10, backgroundColor: red,
    fontFamily: props.headFont, fontSize: 60, lineHeight: 1.15, color: "#FFFFFF", letterSpacing: 3,
    transform: "translateX(" + (-500 * (1 - sChip)) + "px) rotate(-2deg)", boxShadow: "0 12px 26px rgba(0,0,0,0.3)",
  };
  const cardStyle = {
    position: "absolute", left: 0, right: 0, top: 92, bottom: 0, borderRadius: 26, overflow: "hidden",
    border: "1px solid rgba(110,155,255,0.35)", boxShadow: "0 16px 40px rgba(0,0,0,0.35)",
    opacity: sCard, transform: "translateY(" + (30 * (1 - sCard)) + "px)",
  };
  const bgStyle = { position: "absolute", inset: 0, backgroundColor: navy, opacity: 0.88 };
  const rowStyle = { position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 36px" };
  const stepStyleA = { display: "flex", flexDirection: "column", alignItems: "center", gap: 10, width: 330, transform: "scale(" + sA + ")" };
  const stepStyleB = { display: "flex", flexDirection: "column", alignItems: "center", gap: 10, width: 330, transform: "scale(" + sB + ")" };
  const iconRingA = { width: 104, height: 104, borderRadius: 52, display: "flex", alignItems: "center", justifyContent: "center", backgroundColor: red, boxShadow: "0 0 24px rgba(255,45,85,0.55)", transform: "translateY(" + bob + "px)" };
  const iconRingB = { width: 104, height: 104, borderRadius: 52, display: "flex", alignItems: "center", justifyContent: "center", backgroundColor: blue, boxShadow: "0 0 24px rgba(47,107,255,0.6)", transform: "translateY(" + (-bob) + "px)" };
  const labelStyle = { fontFamily: props.bodyFont, fontWeight: 700, fontSize: 30, color: "#FFFFFF", textAlign: "center", letterSpacing: 1, lineHeight: 1.15 };
  const arrowWrap = { position: "relative", width: 200, height: 40, display: "flex", alignItems: "center" };
  const arrowLine = { position: "absolute", left: 0, top: 17, height: 6, width: (200 * arrowGrow) + "px", borderRadius: 3, backgroundImage: "linear-gradient(90deg, " + red + ", " + blue + ")" };
  const arrowHead = { position: "absolute", left: (200 * arrowGrow - 26) + "px", top: 0, opacity: arrowGrow };
  const dotStyle = { position: "absolute", top: 13, left: (travel * 180) + "px", width: 14, height: 14, borderRadius: 7, backgroundColor: "#FFFFFF", opacity: arrowGrow, boxShadow: "0 0 12px #FFFFFF" };

  return (
    <div style={rootStyle}>
      <div style={cardStyle}>
        <div style={bgStyle} />
        <div style={rowStyle}>
          <div style={stepStyleA}>
            <div style={iconRingA}><Icon name="pen" size={56} color="#FFFFFF" strokeWidth={2.4} /></div>
            <div style={labelStyle}>{props.step1}</div>
          </div>
          <div style={arrowWrap}>
            <div style={arrowLine} />
            <div style={dotStyle} />
            <div style={arrowHead}><Icon name="chevron-right" size={40} color={blue} strokeWidth={4} /></div>
          </div>
          <div style={stepStyleB}>
            <div style={iconRingB}><Icon name="soup" size={56} color="#FFFFFF" strokeWidth={2.4} /></div>
            <div style={labelStyle}>{props.step2}</div>
          </div>
        </div>
      </div>
      <div style={chipStyle}>{props.chip}</div>
    </div>
  );
};
