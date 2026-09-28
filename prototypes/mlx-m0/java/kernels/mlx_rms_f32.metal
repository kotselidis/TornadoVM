// M0 spike: MLX's rms_norm kernel (rms_single_row, float, N_READS = 4) run through
// TornadoVM's prebuiltTask. The body is from ml-explore/mlx v0.32.1,
// mlx/backend/metal/kernels/rms_norm.metal (MIT License, Copyright © 2023-2024 Apple Inc.),
// with only the entry point adapted to TornadoVM's prebuilt-kernel ABI:
//   - four implicit leading buffers and a trailing _global_sizes buffer;
//   - array arguments point at the start of TornadoVM's buffer, so the data sits
//     ARRAY_HEADER (16 bytes with compressed oops) further in;
//   - scalars cannot be prebuilt-task arguments, so eps and axis_size come from a
//     FloatArray params = { eps, axis_size }.
#include <metal_stdlib>
using namespace metal;

constant constexpr int ARRAY_HEADER = 16;
constant constexpr int N_READS = 4;
constant constexpr int SIMD_SIZE = 32;

kernel void mlx_rms_f32(
    device long *_kernel_context [[buffer(0)]],
    constant uchar *_constant_region [[buffer(1)]],
    threadgroup uchar *_local_region [[threadgroup(2)]],
    device int *_atomics [[buffer(3)]],
    device uchar *xb [[buffer(4)]],
    device uchar *wb [[buffer(5)]],
    device uchar *ob [[buffer(6)]],
    device uchar *pb [[buffer(7)]],
    uint3 _thread_position_in_grid [[thread_position_in_grid]],
    uint3 _thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 _threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 _local_size [[threads_per_threadgroup]],
    device uint *_global_sizes [[buffer(8)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  const device float *x = (const device float *)(xb + ARRAY_HEADER);
  const device float *w = (const device float *)(wb + ARRAY_HEADER);
  device float *out = (device float *)(ob + ARRAY_HEADER);
  const device float *params = (const device float *)(pb + ARRAY_HEADER);
  const float eps = params[0];
  const uint axis_size = (uint)params[1];
  const uint w_stride = 1;
  const uint gid = _threadgroup_position_in_grid.x;
  const uint lid = _thread_position_in_threadgroup.x;

  // ---- MLX rms_single_row body ----
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  float thread_x[N_READS];
  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[i];
      acc += thread_x[i] * thread_x[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = (lid * N_READS + i < axis_size) ? (float)x[i] : 0;
      acc += thread_x[i] * thread_x[i];
    }
  }
  acc = simd_sum(acc);
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  out += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out[i] = w[w_stride * i] * static_cast<float>(thread_x[i] * local_inv_mean[0]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] = w[w_stride * i] * static_cast<float>(thread_x[i] * local_inv_mean[0]);
      }
    }
  }
}
