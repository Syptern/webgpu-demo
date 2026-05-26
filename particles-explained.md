# WebGPU Particle System — Code Walkthrough

20,000 particles simulated and rendered entirely on the GPU. CPU only handles input and uploads a tiny uniform buffer each frame.

---

## Architecture Overview

```
Each frame:
  CPU → writes uniforms (mouse pos, dt, canvas size)
        ↓
  GPU Compute Pass → updates all 20,000 particle positions/velocities
        ↓
  GPU Render Pass  → draws all 20,000 particles as instanced quads
        ↓
  Screen
```

Two separate GPU pipelines share one particle buffer — compute writes it, render reads it.

---

## Particle Data Layout

Each particle is **32 bytes** (8 × `f32`), stored in a flat GPU storage buffer:

| Field  | Type    | Description                          |
|--------|---------|--------------------------------------|
| `pos`  | `vec2f` | Current position in physical pixels  |
| `vel`  | `vec2f` | Current velocity (pixels/sec)        |
| `home` | `vec2f` | Resting position (spring target)     |
| `seed` | `vec2f` | Random constants for per-particle noise |

`home` is set at init to the particle's starting position and never changes — it's the anchor the spring pulls toward. `seed` is random noise used to give each particle a unique drift pattern.

```js
const STRIDE = 8; // floats per particle
for (let i = 0; i < NUM_PARTICLES; i++) {
  const b = i * STRIDE;
  const x = Math.random() * W;
  const y = Math.random() * H;
  pData[b+0] = x;   pData[b+1] = y;   // pos
  pData[b+2] = 0;   pData[b+3] = 0;   // vel (starts at rest)
  pData[b+4] = x;   pData[b+5] = y;   // home = same as initial pos
  pData[b+6] = Math.random() * 1000;   // seed.x
  pData[b+7] = Math.random() * 1000;   // seed.y
}
```

---

## Uniform Buffers

Two small buffers uploaded from CPU every frame via `device.queue.writeBuffer`.

### Compute uniforms (32 bytes)

```
[mouse.x, mouse.y, dt, time, canvas.w, canvas.h, repelRadius, repelStrength]
```

`repelStrength` is signed: positive = repel, negative = attract. Flipping sign is all that's needed to switch modes on click.

### Render uniforms (16 bytes)

```
[canvas.w, canvas.h, particleSize, time]
```

---

## Compute Shader — Particle Simulation

Runs once per particle per frame. Workgroup size 64, so `ceil(20000 / 64) = 313` workgroups dispatched.

```wgsl
@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&particles)) { return; }
  var p = particles[i];
  // ... simulate ...
  particles[i] = p;
}
```

### Force 1: Mouse repel/attract

```wgsl
let diff = p.pos - u.mouse;      // vector away from mouse
let dist = length(diff);
if (dist < u.repelRadius && dist > 1.0) {
  let t     = 1.0 - dist / u.repelRadius;   // 0 at edge, 1 at center
  let force = t * t * u.repelStrength;       // quadratic falloff
  p.vel += normalize(diff) * force * u.dt;
}
```

`t * t` makes the force much stronger up close than at the edge — feels more physical. When `repelStrength` is negative (click held), `normalize(diff)` points away from mouse but force is negative, so particle accelerates toward mouse.

### Force 2: Home spring

```wgsl
let toHome   = p.home - p.pos;
let homeDist = length(toHome);
if (homeDist > 1.0) {
  p.vel += normalize(toHome) * min(homeDist * 0.6, 120.0) * u.dt;
}
```

Linear spring up to a cap. Particles drift back to their starting positions when undisturbed. Without this, particles would all eventually pile up at random corners.

### Force 3: Organic noise

```wgsl
let nx = sin(p.seed.x * 12.9898 + t * 1.4) * cos(p.seed.y *  4.1414 + t * 0.7);
let ny = cos(p.seed.x * 78.2330 + t * 1.1) * sin(p.seed.y *  5.3219 + t * 0.9);
p.vel += vec2f(nx, ny) * 25.0 * u.dt;
```

Each particle's unique `seed` combined with `time` produces smooth, non-repeating drift. Constants (12.9898, 78.233, etc.) are irrational-looking values that prevent harmonics between particles. Particles at rest appear to breathe and shimmer rather than freeze.

### Damping and speed cap

```wgsl
p.vel *= max(0.0, 1.0 - 2.0 * u.dt);   // exponential decay
if (spd > 800.0) { p.vel = p.vel / spd * 800.0; }
```

Damping is frame-rate independent via `dt`. Speed cap prevents particles from escaping off-screen after a strong repel.

### Edge wrapping

```wgsl
p.pos.x = ((p.pos.x % W) + W) % W;
p.pos.y = ((p.pos.y % H) + H) % H;
```

Double modulo handles negative positions (WGSL `%` can return negative values for negative inputs).

---

## Render Shader — Drawing Particles

No vertex buffer. Each particle is drawn as **two triangles (6 vertices)** using instanced rendering:

```js
rp.draw(6, NUM_PARTICLES); // 6 verts × 20000 instances = 120,000 vertices
```

The vertex shader uses `instance_index` to look up particle data from the storage buffer, and `vertex_index` to pick a corner of a unit quad.

### Vertex shader

```wgsl
var<private> quad: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1),
  vec2f(-1, 1), vec2f(1,-1), vec2f( 1,1),
);

fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> Vout {
  let p     = particles[ii];
  let speed = length(p.vel);
  let sz    = u.particleSize * (1.0 + clamp(speed / 500.0, 0.0, 1.5));  // grow with speed
  let pixel = p.pos + quad[vi] * sz;
  // convert pixel coords to NDC (-1..1 range)
  let ndc = vec2f(
     pixel.x / canvasSize.x *  2.0 - 1.0,
     pixel.y / canvasSize.y * -2.0 + 1.0,  // Y flipped: GPU NDC is +Y up, canvas is +Y down
  );
  ...
}
```

Particles grow slightly at higher speeds — a fast-moving particle appears larger, which reinforces the sense of motion.

### Fragment shader

```wgsl
fn fs(in: Vout) -> @location(0) vec4f {
  let d = distance(in.uv, vec2f(0.5));   // distance from quad center
  if (d > 0.5) { discard; }              // cut quad into a circle

  let core  = 1.0 - smoothstep(0.0, 0.15, d);   // bright hard center
  let glow  = 1.0 - smoothstep(0.05, 0.5, d);   // soft outer glow
  let alpha = core + glow * 0.35;
  ...
}
```

Each particle quad is clipped to a circle via `discard`. Two alpha layers (core + glow) give a bright center with a soft halo.

### Color gradient

```wgsl
let t   = clamp(in.speed / 350.0, 0.0, 1.0);
let t3  = t * t * t;                           // cubic easing
let c0  = vec3f(0.12, 0.14, 0.90);            // indigo  (at rest)
let c1  = vec3f(0.05, 0.85, 1.00);            // cyan    (moving)
let c2  = vec3f(1.00, 1.00, 1.00);            // white   (fast)
let col = mix(mix(c0, c1, t), c2, t3);
```

Linear blend from indigo to cyan, then a cubic boost to white at high speed. Cubic easing keeps particles mostly indigo/cyan, only flashing white when hit hard.

### Additive blending

```js
blend: {
  color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
  alpha: { srcFactor: 'zero',      dstFactor: 'one', operation: 'add' },
}
```

Particles add their color to whatever is already in the framebuffer. Where many particles overlap (especially after a repel burst), they accumulate into bright white clusters. This is what creates the bloom effect without any post-processing.

---

## Frame Loop

```
requestAnimationFrame
  → write compute uniforms (mouse, dt, time, canvas size)
  → write render uniforms  (canvas size, particle size, time)
  → createCommandEncoder
      → beginComputePass  → dispatch 313 workgroups → end
      → beginRenderPass   → draw 6 × 20000          → end
  → queue.submit
  → requestAnimationFrame
```

`dt` is capped at `1/30` second to prevent particles from tunneling through boundaries on a slow frame.

---

## GPU Memory

| Buffer         | Size            | Usage            |
|----------------|-----------------|------------------|
| `particleBuf`  | 640 KB (20k×32B)| STORAGE + COPY_DST |
| `cuBuf`        | 32 B            | UNIFORM + COPY_DST |
| `ruBuf`        | 16 B            | UNIFORM + COPY_DST |

Total GPU memory: ~640 KB. Fits comfortably in L2 cache on most GPUs.

---

## Interaction

| Event         | Effect                                      |
|---------------|---------------------------------------------|
| `mousemove`   | Updates `mouseX/Y` in physical pixels       |
| `mousedown`   | Sets `attract = true` → flips repelStrength to -3500 |
| `mouseup`     | Resets to repel (+2500)                     |
| `mouseleave`  | Moves mouse to (-99999, -99999) — off screen, no force applied |
| `touchmove`   | Same as mousemove                           |
| `touchstart`  | Attract mode on mobile                      |

Physical pixel conversion (`cx * devicePixelRatio`) keeps mouse coordinates in the same space as particle positions, which are stored in physical pixels to match the canvas resolution.
