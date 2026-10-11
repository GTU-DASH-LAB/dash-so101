const Component = ({ item }) => {
  const props = item.props || {};
  const k = props.strength;
  const shade = props.shade;
  const rootStyle = { position: "absolute", inset: 0, backgroundColor: "transparent" };
  const vig = { position: "absolute", left: -240, top: -240, right: -240, bottom: -240, opacity: k, backgroundImage: "radial-gradient(ellipse at 50% 45%, rgba(0,0,0,0) 0%, rgba(0,0,0,0) 48%, " + shade + " 100%)" };
  const top = { position: "absolute", left: 0, right: 0, top: 0, height: 360, opacity: 0.55 * k, backgroundImage: "linear-gradient(180deg, " + shade + " 0%, rgba(0,0,0,0) 100%)" };
  const bottom = { position: "absolute", left: 0, right: 0, bottom: 0, height: 520, opacity: 0.7 * k, backgroundImage: "linear-gradient(0deg, " + shade + " 0%, rgba(0,0,0,0) 100%)" };
  return (
    <div style={rootStyle}>
      <div style={vig} />
      <div style={top} />
      <div style={bottom} />
    </div>
  );
};
