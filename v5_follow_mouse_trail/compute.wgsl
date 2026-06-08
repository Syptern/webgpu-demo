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
  scale      : f32,   // noise frequency (cells per pixel)
  flowSpeed  : f32,   // swirl speed (px/s)
  damping    : f32,   // velocity drag per second
  pull       : f32,   // cohesion toward the cursor
  speedCap   : f32,   // max velocity (px/s)
  morphSpeed : f32,   // how fast the field evolves (Z advance per second)
  mouse      : vec2f, // cursor position in physical pixels
  clusterRadius : f32, // cloud size: no inward pull inside this radius
  jitter     : f32,   // brownian kick to keep the cloud from collapsing
}

@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<uniform>             u         : CU;

// ── 3D Perlin (gradient) noise ───────────────────────────────────────────────
// Hash a 3D lattice point -> f32 in [0,1). `s` shifts the seed for a second,
// independent hash (used to build a 3D gradient direction).
fn hash3(p: vec3f, s: f32) -> f32 {
  var p3 = fract(p * 0.1031 + s);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y + p3.z) * p3.z);
}

// Quintic fade curve (C2 continuous), 3D
fn fade3(t: vec3f) -> vec3f {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Pseudo-random unit vec3 gradient at a lattice point, dotted with the offset.
// Direction sampled uniformly on the sphere from two independent hashes.
fn grad3(ip: vec3f, fp: vec3f) -> f32 {
  let ang = hash3(ip, 0.0) * 6.2831853;     // azimuth
  let z   = hash3(ip, 7.0) * 2.0 - 1.0;     // height on unit sphere
  let r   = sqrt(max(0.0, 1.0 - z * z));
  let g   = vec3f(r * cos(ang), r * sin(ang), z);
  return dot(g, fp);
}

// 3D Perlin noise, range ~[-1, 1]
fn perlin3(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let w = fade3(f);

  // 8 cube corners
  let c000 = grad3(i + vec3f(0.0, 0.0, 0.0), f - vec3f(0.0, 0.0, 0.0));
  let c100 = grad3(i + vec3f(1.0, 0.0, 0.0), f - vec3f(1.0, 0.0, 0.0));
  let c010 = grad3(i + vec3f(0.0, 1.0, 0.0), f - vec3f(0.0, 1.0, 0.0));
  let c110 = grad3(i + vec3f(1.0, 1.0, 0.0), f - vec3f(1.0, 1.0, 0.0));
  let c001 = grad3(i + vec3f(0.0, 0.0, 1.0), f - vec3f(0.0, 0.0, 1.0));
  let c101 = grad3(i + vec3f(1.0, 0.0, 1.0), f - vec3f(1.0, 0.0, 1.0));
  let c011 = grad3(i + vec3f(0.0, 1.0, 1.0), f - vec3f(0.0, 1.0, 1.0));
  let c111 = grad3(i + vec3f(1.0, 1.0, 1.0), f - vec3f(1.0, 1.0, 1.0));

  // trilinear interpolation
  let x00 = mix(c000, c100, w.x);
  let x10 = mix(c010, c110, w.x);
  let x01 = mix(c001, c101, w.x);
  let x11 = mix(c011, c111, w.x);
  let y0  = mix(x00, x10, w.y);
  let y1  = mix(x01, x11, w.y);
  return mix(y0, y1, w.z);
}

// Scalar field: remap perlin from ~[-1,1] to [0,1], then fold to a 0-1-0
// ridge: 1 - |n - 0.5|. Peaks along the n=0.5 contours, valleys at extremes.
fn field(p: vec3f) -> f32 {
  let n01 = perlin3(p) * 0.5 + 0.5;
  return 1.0 - 2.0 * abs(n01 - 0.5);
}

// In-plane (x,y) gradient of the folded field at a fixed Z slice (Z = time).
fn noiseGradient(p: vec2f) -> vec2f {
  let e  = 0.01;
  let z  = u.time * u.morphSpeed;
  let dx = field(vec3f(p.x + e, p.y, z)) - field(vec3f(p.x - e, p.y, z));
  let dy = field(vec3f(p.x, p.y + e, z)) - field(vec3f(p.x, p.y - e, z));
  return vec2f(dx, dy) / (2.0 * e);
}

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&particles)) { return; }
  var p = particles[i];

  // Cursor-local frame so the swirl pattern travels with the mouse.
  let rel = p.pos - u.mouse;
  let d   = max(length(rel), 0.001);

  let toC  = u.mouse - p.pos;
  let fall = clamp(1.0 - d / u.clusterRadius, 0.0, 1.0); // 1 at cursor, 0 at edge

  // Curl swirl, FASTER near the cursor (fall scales the speed up close).
  let g    = noiseGradient(rel * u.scale);
  let flow = vec2f(-g.y, g.x);
  p.vel += flow * u.flowSpeed * (0.5 + 1.5 * fall) * u.dt;

  // Gentle attraction toward the cursor, stronger close in. Bounded so the
  // jitter + swirl keep the cloud spread instead of collapsing to a point.
  p.vel += (toC / d) * u.pull * fall * u.dt;

  // Hard pull-back outside the cloud so strays can't escape.
  if (d > u.clusterRadius) {
    p.vel += (toC / d) * (d - u.clusterRadius) * 4.0 * u.dt;
  }

  // Brownian jitter: random kick (per particle, changing over time) keeps the
  // cloud diffuse so it never settles into the flow's stagnation points.
  let tk  = floor(u.time * 30.0);
  let ang = hash3(vec3f(p.seed.x, p.seed.y, tk), 3.0) * 6.2831853;
  p.vel += vec2f(cos(ang), sin(ang)) * u.jitter * u.dt;

  // Damping
  p.vel *= max(0.0, 1.0 - u.damping * u.dt);

  // Speed cap
  let spd = length(p.vel);
  if (spd > u.speedCap) { p.vel = p.vel / spd * u.speedCap; }

  p.pos += p.vel * u.dt;

  particles[i] = p;
}
