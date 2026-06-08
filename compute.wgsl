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
  morphSpeed : f32,   // how fast the field evolves (Z advance per second)
  mouse      : vec2f, // cursor position in physical pixels
  attract    : f32,   // >0.5 while mouse held down
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

// In-plane (x,y) gradient of the 3D field at a fixed Z slice (Z = time).
// Sliding Z over time makes the field evolve volumetrically in place.
fn noiseGradient(p: vec2f) -> vec2f {
  let e  = 0.01;
  let z  = u.time * u.morphSpeed;
  let dx = perlin3(vec3f(p.x + e, p.y, z)) - perlin3(vec3f(p.x - e, p.y, z));
  let dy = perlin3(vec3f(p.x, p.y + e, z)) - perlin3(vec3f(p.x, p.y - e, z));
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

  // Attract toward the mouse, only while clicking.
  // Quadratic falloff: pull is strongest at the cursor, zero past the radius.
  if (u.attract > 0.5) {
    let toMouse = u.mouse - p.pos;
    let d       = length(toMouse);
    let radius  = 1200.0;
    if (d < radius && d > 1.0) {
      let t = 1.0 - d / radius;             // 1 at cursor, 0 at edge
      p.vel += normalize(toMouse) * t * t * 1000.0 * u.dt;
    }
  }

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
