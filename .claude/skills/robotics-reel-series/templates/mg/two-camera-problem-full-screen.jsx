const Component = ({ item }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const props = item.props || {};
  const red = props.red;
  const blue = props.blue;
  const navy = props.navy;
  const turnAt = props.turnAt;
  const failAt = props.failAt;

  const enter = interpolate(frame, [0, 8], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const exit = interpolate(frame, [durationInFrames - 8, durationInFrames], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const sTitle = spring({ frame: frame - 2, fps, config: { damping: 11, stiffness: 180 } });
  const sScene = spring({ frame: frame - 8, fps, config: { damping: 16, stiffness: 140 } });
  const sArm = spring({ frame: frame - 14, fps, config: { damping: 14, stiffness: 150 } });
  const sCam2 = spring({ frame: frame - 22, fps, config: { damping: 12, stiffness: 170 } });
  const cone = interpolate(frame, [28, 44], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const sFuse = spring({ frame: frame - 40, fps, config: { damping: 16, stiffness: 150 } });
  const turn = interpolate(frame, [turnAt, turnAt + 18], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.back(1.6)) });
  const sTurnText = spring({ frame: frame - turnAt, fps, config: { damping: 16, stiffness: 150 } });
  const sFail = spring({ frame: frame - failAt, fps, config: { damping: 9, stiffness: 200 } });
  const failOn = frame >= failAt ? 1 : 0;
  const l1On = frame >= failAt + 4 ? 1 : 0;
  const l2On = frame >= failAt + 10 ? 1 : 0;
  const shake = frame >= failAt && frame < failAt + 10 ? Math.sin(frame * 3) * 10 : 0;
  const sL1 = spring({ frame: frame - failAt - 4, fps, config: { damping: 14, stiffness: 160 } });
  const sL2 = spring({ frame: frame - failAt - 10, fps, config: { damping: 14, stiffness: 160 } });
  const camAngle = -18 + 46 * turn;
  const wristGlow = 0.5 + 0.5 * Math.sin(frame / 5);

  const bgOpacity = props.transparentBackground ? 0 : 1;
  const rootStyle = { position: "absolute", inset: 0, overflow: "hidden", opacity: enter * (1 - exit), backgroundColor: "transparent", transform: "translateX(" + shake + "px)" };
  const bg = { position: "absolute", inset: 0, backgroundColor: navy, opacity: bgOpacity };
  const grid = { position: "absolute", inset: 0, opacity: bgOpacity, backgroundImage: "linear-gradient(rgba(47,107,255,0.09) 2px, transparent 2px), linear-gradient(90deg, rgba(47,107,255,0.09) 2px, transparent 2px)", backgroundSize: "60px 60px" };
  const glowRed = { position: "absolute", left: -300, top: -240, width: 900, height: 900, borderRadius: 450, opacity: 0.5 * bgOpacity, backgroundImage: "radial-gradient(circle, rgba(255,45,85,0.5) 0%, rgba(255,45,85,0) 70%)" };
  const glowBlue = { position: "absolute", right: -320, bottom: -260, width: 900, height: 900, borderRadius: 450, opacity: 0.5 * bgOpacity, backgroundImage: "radial-gradient(circle, rgba(47,107,255,0.55) 0%, rgba(47,107,255,0) 70%)" };

  const titleStyle = { position: "absolute", left: 80, top: 130, padding: "4px 30px", borderRadius: 14, backgroundColor: red, fontFamily: props.headFont, fontSize: 110, lineHeight: 1.12, color: "#FFFFFF", letterSpacing: 4, transform: "scale(" + sTitle + ") rotate(-2deg)", transformOrigin: "left center", boxShadow: "0 16px 36px rgba(0,0,0,0.4)" };
  const labelStyle = { position: "absolute", left: 96, top: 318, fontFamily: props.monoFont, fontWeight: 700, fontSize: 26, color: "#9DB7FF", letterSpacing: 4, opacity: sScene };

  const scene = { position: "absolute", left: 80, top: 362, width: 920, height: 700, borderRadius: 30, overflow: "hidden", border: "1px solid rgba(110,155,255,0.35)", opacity: sScene, transform: "translateY(" + (30 * (1 - sScene)) + "px)" };
  const sceneBg = { position: "absolute", inset: 0, backgroundColor: "#0B1236", opacity: 0.95 };
  const table = { position: "absolute", left: 40, right: 40, top: 520, height: 6, borderRadius: 3, backgroundColor: "rgba(157,183,255,0.45)" };
  const armWrap = { position: "absolute", left: 120, top: 150, width: 380, height: 380, opacity: sArm, transform: "translateY(" + (40 * (1 - sArm)) + "px)" };
  const wristCam = { position: "absolute", left: 404, top: 214, width: 78, height: 78, borderRadius: 39, backgroundColor: blue, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 0 " + (16 + 18 * wristGlow) + "px rgba(47,107,255,0.8)", opacity: sArm };
  const pen = { position: "absolute", left: 470, top: 494, width: 120, height: 22, borderRadius: 6, backgroundColor: "#FFFFFF", opacity: sArm };
  const penCap = { position: "absolute", left: 470, top: 494, width: 34, height: 22, borderRadius: 6, backgroundColor: red, opacity: sArm };

  const camRig = { position: "absolute", left: 610, top: 70, width: 260, height: 480, transform: "scale(" + sCam2 + ")", transformOrigin: "180px 60px" };
  const stand = { position: "absolute", left: 172, top: 100, width: 14, height: 350, borderRadius: 7, backgroundColor: "rgba(157,183,255,0.6)" };
  const standFoot = { position: "absolute", left: 120, top: 444, width: 120, height: 12, borderRadius: 6, backgroundColor: "rgba(157,183,255,0.6)" };
  const headWrap = { position: "absolute", left: 0, top: 0, width: 260, height: 480, transform: "rotate(" + camAngle + "deg)", transformOrigin: "180px 60px" };
  const coneStyle = { position: "absolute", left: -330, top: 30, width: 510, height: 460, opacity: 0.85 * cone };
  const camHead = { position: "absolute", left: 136, top: 22, width: 88, height: 78, borderRadius: 18, backgroundColor: red, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 0 22px rgba(255,45,85,0.7)" };

  const fuse = { position: "absolute", left: 40, right: 40, top: 586, display: "flex", alignItems: "center", gap: 14, opacity: sFuse * (1 - turn), transform: "translateY(" + (16 * (1 - sFuse)) + "px)" };
  const fuseText = { fontFamily: props.bodyFont, fontWeight: 700, fontSize: 36, color: "#FFFFFF" };
  const turnLine = { position: "absolute", left: 40, right: 40, top: 586, display: "flex", alignItems: "center", gap: 14, opacity: sTurnText * (1 - failOn), transform: "translateY(" + (16 * (1 - sTurnText)) + "px)" };
  const turnTextStyle = { fontFamily: props.bodyFont, fontWeight: 700, fontSize: 36, color: "#FFD3DC" };

  const stampWrap = { position: "absolute", left: 140, top: 560, width: 800, height: 220, display: "flex", alignItems: "center", justifyContent: "center", opacity: failOn, transform: "scale(" + (2.2 - 1.2 * sFail) + ") rotate(-8deg)" };
  const stamp = { display: "flex", alignItems: "center", gap: 16, padding: "10px 34px", borderRadius: 18, border: "8px solid " + red, backgroundColor: "rgba(6,10,31,0.85)", boxShadow: "0 0 40px rgba(255,45,85,0.6)" };
  const stampText = { fontFamily: props.headFont, fontSize: 110, lineHeight: 1.1, color: red, letterSpacing: 4 };

  const lesson = { position: "absolute", left: 80, top: 1350, width: 920, display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 14 };
  const l1 = { padding: "6px 26px", borderRadius: 10, backgroundColor: blue, fontFamily: props.headFont, fontSize: 70, color: "#FFFFFF", letterSpacing: 3, opacity: l1On, transform: "translateX(" + (-900 * (1 - sL1)) + "px)" };
  const l2 = { padding: "6px 26px", borderRadius: 10, backgroundColor: red, fontFamily: props.headFont, fontSize: 70, color: "#FFFFFF", letterSpacing: 3, opacity: l2On, transform: "translateX(" + (900 * (1 - sL2)) + "px)" };

  return (
    <div style={rootStyle}>
      <div style={bg} />
      <div style={grid} />
      <div style={glowRed} />
      <div style={glowBlue} />
      <div style={labelStyle}>{props.label}</div>
      <div style={scene}>
        <div style={sceneBg} />
        <div style={table} />
        <div style={armWrap}>
          <svg width="380" height="380" viewBox="0 0 380 380">
            <rect x="40" y="330" width="120" height="40" rx="10" fill="#E8EEFF" />
            <rect x="80" y="300" width="40" height="40" rx="8" fill="#C9D3FF" />
            <line x1="100" y1="310" x2="150" y2="110" stroke="#E8EEFF" strokeWidth="34" strokeLinecap="round" />
            <circle cx="150" cy="110" r="24" fill="#9DB7FF" />
            <line x1="150" y1="110" x2="300" y2="170" stroke="#E8EEFF" strokeWidth="28" strokeLinecap="round" />
            <circle cx="300" cy="170" r="20" fill="#9DB7FF" />
            <line x1="300" y1="170" x2="318" y2="300" stroke="#2F6BFF" strokeWidth="16" strokeLinecap="round" />
            <line x1="300" y1="170" x2="282" y2="300" stroke="#2F6BFF" strokeWidth="16" strokeLinecap="round" />
          </svg>
        </div>
        <div style={wristCam}><Icon name="camera" size={40} color="#FFFFFF" strokeWidth={2.4} /></div>
        <div style={pen} />
        <div style={penCap} />
        <div style={camRig}>
          <div style={stand} />
          <div style={standFoot} />
          <div style={headWrap}>
            <svg width="510" height="460" viewBox="0 0 510 460" style={coneStyle}>
              <polygon points="500,40 20,300 120,440" fill="rgba(255,45,85,0.16)" stroke="rgba(255,45,85,0.7)" strokeWidth="3" strokeDasharray="12 10" />
            </svg>
            <div style={camHead}><Icon name="video" size={44} color="#FFFFFF" strokeWidth={2.4} /></div>
          </div>
        </div>
        <div style={fuse}><Icon name="layers" size={40} color="#9DB7FF" strokeWidth={2.4} /><div style={fuseText}>{props.fuse}</div></div>
        <div style={turnLine}><Icon name="rotate-cw" size={40} color={red} strokeWidth={2.6} /><div style={turnTextStyle}>{props.turnText}</div></div>
      </div>
      <div style={stampWrap}><div style={stamp}><Icon name="x" size={110} color={red} strokeWidth={4} /><div style={stampText}>{props.stamp}</div></div></div>
      <div style={titleStyle}>{props.title}</div>
      <div style={lesson}>
        <div style={l1}>{props.lesson1}</div>
        <div style={l2}>{props.lesson2}</div>
      </div>
    </div>
  );
};
