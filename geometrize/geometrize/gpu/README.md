# GPU Evaluator Notes

## What Is Accelerated
- Candidate shape rasterization (in compute shader).
- Candidate color estimation for alpha blending.
- Candidate error scoring against target/current textures.
- Best-candidate reduction on GPU (single-index readback).

## Memory Layout
- `target` bitmap: `image2D` RGBA8 (`binding=0`).
- `current` bitmap: `image2D` RGBA8 (`binding=1`).
- Candidate shapes: `std430` SSBO (`binding=0`), packed as:
  - `int type`
  - `int alpha`
  - `int reserved0`
  - `int reserved1`
  - `vec4 p0`
  - `vec4 p1`
- Candidate scores: `std430` SSBO (`binding=1`), `float scores[]`.
- Candidate colors: `std430` SSBO (`binding=2`), `vec4 colors[]`.
- Reduction output: `std430` SSBO (`binding=1` in reducer), `{uint bestIndex; float bestScore; ...}`.

## Fallback Rules
- Any custom energy function forces CPU evaluation.
- Missing OpenGL 4.3 context or unavailable GLEW/OpenGL symbols forces CPU.
- Unsupported shape types (`line`, `polyline`, `quadratic_bezier`) force CPU.
- Fallback is logged once from `Model`.

## Performance Considerations
- Batch evaluation dispatches one workgroup per candidate.
- Workgroup-shared reductions avoid global atomics for per-candidate accumulation.
- Only the reduced winner and winning color are read back to CPU.
- Textures are reused and updated each step; no per-candidate texture uploads.

## Expected Improvements
- The biggest gain is when many candidates are evaluated per step and images are large.
- Typical speedups are expected in the random candidate phase.
- Hill-climb iterations also use GPU scoring, but overall gain depends on mutation count and shape mix.
