const Component = ({ item }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const props = item.props || {};
  const red = props.red;
  const blue = props.blue;
  const navy = props.navy;

  const s = spring({ frame: frame, fps, config: { damping: 9, stiffness: 200 } });
  const sSub = spring({ frame: frame - 8, fps, config: { damping: 16, stiffness: 150 } });
  const exit = interpolate(frame, [durationInFrames - 8, durationInFrames], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const wobble = Math.sin(frame / 4) * 6;
  const caret = Math.floor(frame / 8) % 2 === 0 ? 1 : 0;

  const rootStyle = { position: "absolute", inset: 0, backgroundColor: "transparent", opacity: 1 - exit, transform: "translateY(" + (30 * exit) + "px)" };
  const col = { position: "absolute", left: 0, top: 0, display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 14 };
  const chip = { display: "flex", alignItems: "center", gap: 16, padding: "6px 30px", borderRadius: 14, backgroundColor: red, fontFamily: props.headFont, fontSize: 104, lineHeight: 1.1, color: "#FFFFFF", letterSpacing: 4, transform: "scale(" + s + ") rotate(-2deg)", transformOrigin: "left center", boxShadow: "0 14px 30px rgba(0,0,0,0.35)" };
  const qWrap = { transform: "rotate(" + wobble + "deg)" };
  const sub = { position: "relative", display: "flex", alignItems: "center", gap: 12, padding: "10px 22px", borderRadius: 12, overflow: "hidden", opacity: sSub, transform: "translateX(" + (-40 * (1 - sSub)) + "px)" };
  const subBg = { position: "absolute", inset: 0, backgroundColor: navy, opacity: 0.88 };
  const subBar = { position: "relative", width: 8, height: 34, borderRadius: 4, backgroundColor: blue };
  const subText = { position: "relative", fontFamily: props.monoFont, fontWeight: 700, fontSize: 34, color: "#FFFFFF" };
  const caretStyle = { position: "relative", width: 16, height: 34, backgroundColor: "#FFFFFF", opacity: caret };

  return (
    <div style={rootStyle}>
      <div style={col}>
        <div style={chip}>{props.title}<div style={qWrap}><Icon name="circle-help" size={80} color="#FFFFFF" strokeWidth={2.6} /></div></div>
        <div style={sub}><div style={subBg} /><div style={subBar} /><div style={subText}>{props.sub}</div><div style={caretStyle} /></div>
      </div>
    </div>
  );
};
