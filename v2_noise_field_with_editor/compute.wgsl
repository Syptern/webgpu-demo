struct Particle {
  pos  : vec2f,
  vel  : vec2f,
  home : vec2f,
  seed : vec2f,
}

struct CU {
  canvasSize : vec2f,
  dt         : f32,
  time       : f32,
  scale      : f32,   // field frequency (cells per pixel)
  flowSpeed  : f32,   // how fast particles ride the field
  damping    : f32,   // velocity drag per second
  sink       : f32,   // 0 = pure flow, >0 drain to minima, <0 bloom out
  speedCap   : f32,   // max velocity (px/s)
}

@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<uniform>             u         : CU;

// ── Perlin (gradient) noise ──────────────────────────────────────────────────
fn hash(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

// Quintic fade curve (C2 continuous)
fn fade(t: vec2f) -> vec2f {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Pseudo-random unit gradient at a lattice point, dotted with the offset
fn grad(ip: vec2f, fp: vec2f) -> f32 {
  let ang = hash(ip) * 6.2831853;
  return dot(vec2f(cos(ang), sin(ang)), fp);
}

// 2D Perlin noise, range ~[-1, 1]
fn perlin(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let w = fade(f);
  let a = grad(i,                      f);
  let b = grad(i + vec2f(1.0, 0.0), f - vec2f(1.0, 0.0));
  let c = grad(i + vec2f(0.0, 1.0), f - vec2f(0.0, 1.0));
  let d = grad(i + vec2f(1.0, 1.0), f - vec2f(1.0, 1.0));
  return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

// Gradient of the noise field (which way is uphill), via central difference
fn noiseGradient(p: vec2f) -> vec2f {
  let e  = 0.01;
  let dx = perlin(p + vec2f(e, 0.0)) - perlin(p - vec2f(e, 0.0));
  let dy = perlin(p + vec2f(0.0, e)) - perlin(p - vec2f(0.0, e));
  return vec2f(dx, dy) / (2.0 * e);
}

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&particles)) { return; }
  var p = particles[i];

  // Sample the Perlin field at this particle's position
  let sp = p.pos * u.scale;

  // Flow ALONG the field: rotate gradient 90deg -> divergence-free curl.
  // `sink` blends in -gradient to drain toward minima (or bloom outward).
  let g    = noiseGradient(sp);
  let flow = vec2f(-g.y, g.x) - g * u.sink;
  p.vel += flow * u.flowSpeed * u.dt;

  // Damping
  p.vel *= max(0.0, 1.0 - u.damping * u.dt);

  // Speed cap
  let spd = length(p.vel);
  if (spd > u.speedCap) { p.vel = p.vel / spd * u.speedCap; }

  p.pos += p.vel * u.dt;

  // Wrap at canvas edges
  let W = u.canvasSize.x;
  let H = u.canvasSize.y;
  p.pos.x = ((p.pos.x % W) + W) % W;
  p.pos.y = ((p.pos.y % H) + H) % H;

  particles[i] = p;
}
