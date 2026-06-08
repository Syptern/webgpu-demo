struct Particle {
  pos  : vec2f,
  vel  : vec2f,
  home : vec2f,
  seed : vec2f,
}

struct RU {
  canvasSize   : vec2f,
  particleSize : f32,
  brightness   : f32,
  time         : f32,
}

@group(0) @binding(0) var<storage, read> particles : array<Particle>;
@group(0) @binding(1) var<uniform>       u         : RU;

var<private> quad: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
  vec2f(-1.0,  1.0), vec2f(1.0, -1.0), vec2f( 1.0, 1.0),
);

struct Vout {
  @builtin(position) pos   : vec4f,
  @location(0)       uv    : vec2f,
  @location(1)       speed : f32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> Vout {
  let p     = particles[ii];
  let q     = quad[vi];
  let speed = length(p.vel);
  let sz    = u.particleSize * (1.0 + clamp(speed / 500.0, 0.0, 1.5));
  let pixel = p.pos + q * sz;
  let ndc   = vec2f(
     pixel.x / u.canvasSize.x *  2.0 - 1.0,
     pixel.y / u.canvasSize.y * -2.0 + 1.0,
  );
  var out: Vout;
  out.pos   = vec4f(ndc, 0.0, 1.0);
  out.uv    = (q + 1.0) * 0.5;
  out.speed = speed;
  return out;
}

@fragment
fn fs(in: Vout) -> @location(0) vec4f {
  let d = distance(in.uv, vec2f(0.5));
  if (d > 0.5) { discard; }

  let core  = 1.0 - smoothstep(0.0, 0.15, d);
  let glow  = 1.0 - smoothstep(0.05, 0.5, d);
  let alpha = core + glow * 0.35;

  // Indigo at rest -> cyan -> white at max speed
  let t   = clamp(in.speed / 350.0, 0.0, 1.0);
  let t3  = t * t * t;
  let c0  = vec3f(0.12, 0.14, 0.90);
  let c1  = vec3f(0.05, 0.85, 1.00);
  let c2  = vec3f(1.00, 1.00, 1.00);
  let col = mix(mix(c0, c1, t), c2, t3);

  return vec4f(col, alpha * u.brightness);
}
