// Prompt-window EXL3 MoE, second generation: launchers, the input conversion and the deterministic slot sum.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "prompt_moe.cuh"

namespace {

// xh [R, D] = fp16(x * 2^-2), x bf16 or fp16 rows of stride xs; zero [R, D] fp32 (optional) = 0
template <typename T>
__global__ void to_half_kernel(const T* __restrict__ x, int xs, half* __restrict__ xh, int D, float* __restrict__ zero) {
    const int r = blockIdx.y;
    const int d = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (d >= D) return;
    if (zero) {
        float4* z = reinterpret_cast<float4*>(zero + (size_t)r * D + d);
        z[0] = make_float4(0.f, 0.f, 0.f, 0.f);
        z[1] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    const T* p = x + (size_t)r * xs + d;
    half o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float v;
        if constexpr (sizeof(T) == 2 && std::is_same<T, __nv_bfloat16>::value) v = __bfloat162float(p[j]);
        else v = __half2float(p[j]);
        o[j] = __float2half_rn(v * tf_exl3p::X_SCALE);
    }
    *reinterpret_cast<uint4*>(xh + (size_t)r * D + d) = *reinterpret_cast<uint4*>(o);
}

// out[r] = sum over the routed slots in slot order of wts * y (y already * 1, fp32): the fixed-order summed path
__global__ void slot_sum_kernel(const float* __restrict__ y, const int* __restrict__ pick,
                                const float* __restrict__ wts, float* __restrict__ out, int D, int slots, int E) {
    const int r = blockIdx.y;
    const int d = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (d >= D) return;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int k = 0; k < slots; ++k) {
        const int p = r * slots + k;
        const int e = pick[p];
        if (e < 0 || e >= E) continue;
        const float w = wts[p];
        const float4 v = *reinterpret_cast<const float4*>(y + (size_t)p * D + d);
        acc.x = fmaf(w, v.x, acc.x); acc.y = fmaf(w, v.y, acc.y);
        acc.z = fmaf(w, v.z, acc.z); acc.w = fmaf(w, v.w, acc.w);
    }
    *reinterpret_cast<float4*>(out + (size_t)r * D + d) = acc;
}

// grid[r] = 2^(e - 24) with 2^e >= (1 + 2^-6) * sum over the row's routed slots of |w| * ||xd|| * colmax[e]: a bound on
// the sum of |terms| in any column of the row (Cauchy-Schwarz with each down matrix's largest column norm, H128, svh
// and 1/sqrt(128) included; the 2^-6 covers the fp32 roundings of y and of the bound). Every term rounded to the grid
// is an integer multiple of it and every partial sum stays below 2^e = 2^24 grid, so the fp32 red.adds are exact and
// their order cannot change a bit. One thread a row, the slots and the norm's column groups in a fixed order: a row's
// grid is its own (window and run independent).
__global__ void row_grid_kernel(const float* __restrict__ nrm, const int* __restrict__ pick, const float* __restrict__ wts,
                                const float* __restrict__ colmax, float* __restrict__ grid, int R, int slots, int E, int G) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    float b = 0.f;
    for (int k = 0; k < slots; ++k) {
        const int p = r * slots + k;
        const int e = pick[p];
        if (e < 0 || e >= E) continue;
        float n2 = 0.f;
        for (int j = 0; j < G; ++j) n2 = __fadd_rn(n2, nrm[(size_t)p * G + j]);
        b = __fmaf_rn(__fmul_rn(fabsf(wts[p]), __fsqrt_rn(n2)), colmax[e], b);
    }
    b = __fmul_rn(b, 1.015625f);
    int ex = 0;
    if (b > 0.f && isfinite(b)) frexpf(b, &ex);     // b < 2^ex
    ex = max(min(ex, 100), -100);
    grid[r] = ldexpf(1.f, ex - 24);
}

// colmax: one warp a (expert, 128-column block) of W_q [n, I, D] fp16: each row's 128 values through H128
// (registers + lane shuffles, fixed order), squares summed over the rows in row order, then * svh^2 and the block's
// max (order-free). out [n, D / 128]: the down map's largest squared column norm a block (before 1/128).
__global__ void colmax_kernel(const half* __restrict__ w, const half* __restrict__ svh, float* __restrict__ out, int I,
                              int D) {
    const int b = blockIdx.x, e = blockIdx.y, lane = threadIdx.x;
    const half* we = w + (size_t)e * I * D + b * 128 + 4 * lane;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int i = 0; i < I; ++i) {
        const uint2 raw = *reinterpret_cast<const uint2*>(we + (size_t)i * D);
        const float2 p = tf_exl3p::h2f(raw.x), q = tf_exl3p::h2f(raw.y);
        float v[4] = {p.x, p.y, q.x, q.y};
        const float a0 = __fadd_rn(v[0], v[1]), b0 = __fsub_rn(v[0], v[1]);
        const float c0 = __fadd_rn(v[2], v[3]), d0 = __fsub_rn(v[2], v[3]);
        v[0] = __fadd_rn(a0, c0); v[1] = __fadd_rn(b0, d0); v[2] = __fsub_rn(a0, c0); v[3] = __fsub_rn(b0, d0);
#pragma unroll
        for (int m = 1; m < 32; m <<= 1)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
                v[j] = (lane & m) ? __fsub_rn(o, v[j]) : __fadd_rn(v[j], o);
            }
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = __fmaf_rn(v[j], v[j], acc[j]);
    }
    float mx = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float s = __half2float(svh[(size_t)e * D + b * 128 + 4 * lane + j]);
        mx = fmaxf(mx, __fmul_rn(acc[j], __fmul_rn(s, s)));
    }
    for (int m = 16; m; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, m));
    if (lane == 0) out[(size_t)e * (D / 128) + b] = mx;
}

}  // namespace

static const half* hp(const at::Tensor& t) { return reinterpret_cast<const half*>(t.data_ptr()); }

void p2_to_half_cuda(const at::Tensor& x, int64_t xs, at::Tensor& xh, int64_t R, int64_t D, at::Tensor* zero) {
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)((D / 8 + 127) / 128), (unsigned)R);
    float* z = zero ? zero->data_ptr<float>() : nullptr;
    if (x.scalar_type() == at::kBFloat16)
        to_half_kernel<__nv_bfloat16><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)xs, reinterpret_cast<half*>(xh.data_ptr()), (int)D, z);
    else
        to_half_kernel<half><<<grid, 128, 0, stream>>>(hp(x), (int)xs, reinterpret_cast<half*>(xh.data_ptr()), (int)D, z);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void p2_gate_up_cuda(const at::Tensor& xh, const at::Tensor& tp0, const at::Tensor& tp1, const at::Tensor& k2,
                     const at::Tensor& suh0, const at::Tensor& suh1, const at::Tensor& svh0, const at::Tensor& svh1,
                     const at::Tensor& suh_d, const at::Tensor& uids, const at::Tensor& ucount,
                     const at::Tensor& members, const at::Tensor& work, const at::Tensor& items_t, at::Tensor& xd, at::Tensor* nrm,
                     int64_t D, int64_t I, int64_t slots, int64_t cb, int64_t items, int64_t k2max, double limit, int64_t act_mode) {
    tf_exl3p::GuArgs g;
    g.xh = hp(xh); g.tp0 = tp0.data_ptr<int64_t>(); g.tp1 = tp1.data_ptr<int64_t>(); g.k2 = k2.data_ptr<int>();
    g.suh0 = hp(suh0); g.suh1 = hp(suh1); g.svh0 = hp(svh0); g.svh1 = hp(svh1); g.suh_d = hp(suh_d);
    g.uids = uids.data_ptr<int>(); g.ucount = ucount.data_ptr<int>(); g.members = members.data_ptr<int>();
    g.work = work.data_ptr<int>(); g.xd = reinterpret_cast<half*>(xd.data_ptr()); g.nrm = nrm ? nrm->data_ptr<float>() : nullptr;
    g.items = reinterpret_cast<const int2*>(items_t.data_ptr<int>());
    g.D = (int)D; g.I = (int)I; g.maxm = (int)members.size(1); g.slots = (int)slots;
    g.limit = (float)limit; g.act_mode = (int)act_mode;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (cb == 0) tf_exl3p::gu_launch<0>(g, (int)items, (int)k2max, stream);
    else if (cb == 1) tf_exl3p::gu_launch<1>(g, (int)items, (int)k2max, stream);
    else tf_exl3p::gu_launch<2>(g, (int)items, (int)k2max, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void p2_down_cuda(const at::Tensor& xd, const at::Tensor& tp, const at::Tensor& k2, const at::Tensor& svh,
                  const at::Tensor& uids, const at::Tensor& ucount, const at::Tensor& members, const at::Tensor& work,
                  const at::Tensor& items_t, const at::Tensor* wts, const at::Tensor* grid, at::Tensor& out, int64_t D, int64_t I,
                  int64_t slots, int64_t cb, int64_t items, int64_t k2max, int64_t rows, int64_t out_budget) {
    tf_exl3p::DnArgs g;
    g.rows = (int)rows;
    if (out_budget > 0) g.out_budget = out_budget;
    g.xd = hp(xd); g.tp = tp.data_ptr<int64_t>(); g.k2 = k2.data_ptr<int>(); g.svh = hp(svh);
    g.uids = uids.data_ptr<int>(); g.ucount = ucount.data_ptr<int>(); g.members = members.data_ptr<int>();
    g.work = work.data_ptr<int>(); g.items = reinterpret_cast<const int2*>(items_t.data_ptr<int>()); g.wts = wts ? wts->data_ptr<float>() : nullptr;
    g.grid = grid ? grid->data_ptr<float>() : nullptr; g.out = out.data_ptr<float>();
    g.D = (int)D; g.I = (int)I; g.maxm = (int)members.size(1); g.slots = (int)slots; g.summed = wts ? 1 : 0;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (cb == 0) tf_exl3p::dn_launch<0>(g, (int)items, (int)k2max, stream);
    else if (cb == 1) tf_exl3p::dn_launch<1>(g, (int)items, (int)k2max, stream);
    else tf_exl3p::dn_launch<2>(g, (int)items, (int)k2max, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void p2_slot_sum_cuda(const at::Tensor& y, const at::Tensor& pick, const at::Tensor& wts, at::Tensor& out, int64_t R,
                      int64_t D, int64_t slots, int64_t E) {
    dim3 grid((unsigned)((D / 4 + 127) / 128), (unsigned)R);
    slot_sum_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        y.data_ptr<float>(), pick.data_ptr<int>(), wts.data_ptr<float>(), out.data_ptr<float>(), (int)D, (int)slots,
        (int)E);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void p2_items_cuda(const at::Tensor& work, const at::Tensor& ucount, at::Tensor& items, int64_t n) {
    tf_exl3p::item_table_kernel<<<(unsigned)((n + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        work.data_ptr<int>(), ucount.data_ptr<int>(), reinterpret_cast<int2*>(items.data_ptr<int>()), (int)n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void p2_row_grid_cuda(const at::Tensor& nrm, const at::Tensor& pick, const at::Tensor& wts, const at::Tensor& colmax,
                      at::Tensor& grid, int64_t R, int64_t slots, int64_t E, int64_t G) {
    if (R <= 0) return;
    row_grid_kernel<<<(unsigned)((R + 127) / 128), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        nrm.data_ptr<float>(), pick.data_ptr<int>(), wts.data_ptr<float>(), colmax.data_ptr<float>(),
        grid.data_ptr<float>(), (int)R, (int)slots, (int)E, (int)G);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void p2_colmax_cuda(const at::Tensor& w, const at::Tensor& svh, at::Tensor& out, int64_t n, int64_t I, int64_t D) {
    if (n <= 0) return;
    colmax_kernel<<<dim3((unsigned)(D / 128), (unsigned)n), 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        hp(w), hp(svh), out.data_ptr<float>(), (int)I, (int)D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
