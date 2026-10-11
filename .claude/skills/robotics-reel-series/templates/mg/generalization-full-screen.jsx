const Component = ({ item }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const props = item.props || {};
  const red = props.red;
  const blue = props.blue;
  const navy = props.navy;

  const enter = interpolate(frame, [0, 6], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const exit = interpolate(frame, [durationInFrames - 6, durationInFrames], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const sTitle = spring({ frame: frame - 1, fps, config: { damping: 12, stiffness: 180 } });
  const sFrame = spring({ frame: frame - 4, fps, config: { damping: 15, stiffness: 160 } });
  const sL1 = spring({ frame: frame - 14, fps, config: { damping: 16, stiffness: 150 } });
  const sL2 = spring({ frame: frame - 24, fps, config: { damping: 12, stiffness: 170 } });
  const blink = Math.floor(frame / 10) % 2 === 0 ? 1 : 0.3;
  const sweep = interpolate(frame, [24, 70], [-20, 120], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  const bgOpacity = props.transparentBackground ? 0 : 1;
  const rootStyle = { position: "absolute", inset: 0, overflow: "hidden", opacity: enter * (1 - exit), backgroundColor: "transparent" };
  const bg = { position: "absolute", inset: 0, backgroundColor: navy, opacity: bgOpacity };
  const grid = { position: "absolute", inset: 0, opacity: bgOpacity, backgroundImage: "linear-gradient(rgba(47,107,255,0.09) 2px, transparent 2px), linear-gradient(90deg, rgba(47,107,255,0.09) 2px, transparent 2px)", backgroundSize: "60px 60px" };
  const glowBlue = { position: "absolute", left: 90, top: 260, width: 900, height: 900, borderRadius: 450, opacity: 0.6 * bgOpacity, backgroundImage: "radial-gradient(circle, rgba(47,107,255,0.5) 0%, rgba(47,107,255,0) 70%)" };

  const title = { position: "absolute", left: 80, top: 160, padding: "4px 28px", borderRadius: 14, backgroundColor: blue, fontFamily: props.headFont, fontSize: 104, lineHeight: 1.12, color: "#FFFFFF", letterSpacing: 3, transform: "scale(" + sTitle + ") rotate(-2deg)", transformOrigin: "left center", boxShadow: "0 16px 36px rgba(0,0,0,0.4)" };
  const frameBorder = { position: "absolute", left: 170, top: 420, width: 740, height: 560, borderRadius: 30, border: "5px solid " + blue, boxShadow: "0 0 40px rgba(47,107,255,0.6)", opacity: sFrame, transform: "scale(" + (0.9 + 0.1 * sFrame) + ")" };
  const tab = { position: "absolute", left: 200, top: 392, display: "flex", alignItems: "center", gap: 12, padding: "8px 18px", borderRadius: 12, backgroundColor: navy, border: "2px solid " + blue, opacity: sFrame };
  const dot = { width: 14, height: 14, borderRadius: 7, backgroundColor: red, opacity: blink };
  const tabText = { fontFamily: props.monoFont, fontWeight: 700, fontSize: 26, color: "#FFFFFF", letterSpacing: 2 };
  const l1 = { position: "absolute", left: 80, right: 80, top: 1010, textAlign: "center", fontFamily: props.bodyFont, fontWeight: 700, fontSize: 42, color: "#C9D3FF", opacity: sL1, transform: "translateY(" + (16 * (1 - sL1)) + "px)" };
  const l2Wrap = { position: "absolute", left: 0, right: 0, top: 1350, display: "flex", justifyContent: "center" };
  const l2 = { position: "relative", overflow: "hidden", padding: "6px 30px", borderRadius: 12, backgroundColor: red, fontFamily: props.headFont, fontSize: 84, lineHeight: 1.12, color: "#FFFFFF", letterSpacing: 4, transform: "scale(" + sL2 + ")" };
  const shine = { position: "absolute", top: 0, bottom: 0, width: 140, left: sweep + "%", backgroundImage: "linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.35) 50%, rgba(255,255,255,0) 100%)" };

  return (
    <div style={rootStyle}>
      <div style={bg} />
      <div style={grid} />
      <div style={glowBlue} />
      <div style={frameBorder} />
      <div style={tab}><div style={dot} /><div style={tabText}>{props.tab}</div></div>
      <div style={title}>{props.title}</div>
      <div style={l1}>{props.line1}</div>
      <div style={l2Wrap}><div style={l2}><div style={shine} />{props.line2}</div></div>
    </div>
  );
};
