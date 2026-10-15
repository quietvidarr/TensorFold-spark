// Hyper-connection fp16 matmuls at decode row counts: y = fp16(x) @ w.T, the arithmetic of exl3_mm._f16_mm (BK 64
// steps in order, each 4 mma.m16n8k16 k16 steps in order, fp32 accumulators), so every output has its bits.
// A warp owns 16 columns of one K slice for up to 64 rows (row tiles share each B fragment: W is read once a call);
// W streams through a per-warp cp.async ring (no L2 hints: illegal on GB10), read back with ldmatrix.
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int WARPS = 4;
constexpr int CHUNK = 64;                         // k per stage (Triton's BK)
constexpr int STAGE_BYTES = 16 * CHUNK * 2;       // 16 rows of W x 64 k fp16 = 2 KB

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// row r (0..15), 16-byte chunk c (0..7) of a stage: XOR swizzle so ldmatrix's 8 rows hit 8 banks groups
__device__ __forceinline__ int swz(int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); }

__device__ __forceinline__ uint32_t x_pair(const __nv_bfloat16* x, bool ok) {
    if (!ok) return 0u;                                           // Triton's masked rows: other=0.0
    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(x);
    const half2 h = __floats2half2_rn(__bfloat162float(v.x), __bfloat162float(v.y));   // x.to(fp16): RN from exact fp32
    return *reinterpret_cast<const uint32_t*>(&h);
}

// MT row tiles (16 rows each) a pass; F32: fp32 slices [SK, M, N] (o_stride ignored), else bf16 [M, o_stride].
template <int MT, bool F32>
__global__ void __launch_bounds__(WARPS * 32) hc_gemv_kernel(const __nv_bfloat16* __restrict__ X, const half* __restrict__ W,
                                                             void* __restrict__ OUT, int M, int N, int K, int KS,
                                                             int x_stride, int o_stride, int tiles, int items) {
    constexpr int S = 4;                                          // stages in flight a warp
    __shared__ __align__(128) unsigned char smem[WARPS][S][STAGE_BYTES];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int item = blockIdx.x * WARPS + warp;
    if (item >= items) return;
    const int tile = item % tiles, slice = item / tiles;
    const int n0 = tile * 16, k0 = slice * KS, steps = KS / CHUNK;
    const int g = lane >> 2, t = lane & 3;
    unsigned char* ring = &smem[warp][0][0];

    // this lane's 4 cp.async chunks of a stage: rows (lane >> 3) + 4 q, 16-byte chunk lane & 7
    const half* src[4];
    int dst[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int r = (lane >> 3) + 4 * q, c = lane & 7;
        const int n = min(n0 + r, N - 1);                         // rows past N: any valid row (outputs dropped)
        src[q] = W + (size_t)n * K + k0 + c * 8;
        dst[q] = swz(r, c);
    }
    auto issue = [&](int s) {
        if (s < steps) {
            unsigned char* st = ring + (s % S) * STAGE_BYTES;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const unsigned sa = (unsigned)__cvta_generic_to_shared(st + dst[q]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(src[q] + s * CHUNK));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    // ldmatrix.x4 addresses: matrices (n 0-7, k 0-7), (n 0-7, k 8-15), (n 8-15, k 0-7), (n 8-15, k 8-15)
    const int lr = (lane & 7) + ((lane >> 4) << 3), lc = (lane >> 3) & 1;

    for (int m0 = 0; m0 < M; m0 += 16 * MT) {
        float acc[MT][2][4];
#pragma unroll
        for (int r = 0; r < MT; ++r)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[r][h][c] = 0.f;
        const __nv_bfloat16* xr[MT][2];
        bool ok[MT][2];
#pragma unroll
        for (int r = 0; r < MT; ++r) {
            const int a0 = m0 + 16 * r + g, a1 = a0 + 8;
            ok[r][0] = a0 < M;
            ok[r][1] = a1 < M;
            xr[r][0] = X + (size_t)(ok[r][0] ? a0 : 0) * x_stride + k0 + 2 * t;
            xr[r][1] = X + (size_t)(ok[r][1] ? a1 : 0) * x_stride + k0 + 2 * t;
        }
#pragma unroll
        for (int s = 0; s < S - 1; ++s) issue(s);
#pragma unroll 1
        for (int s = 0; s < steps; ++s) {
            issue(s + S - 1);
            asm volatile("cp.async.wait_group %0;\n" ::"n"(S - 1));
            __syncwarp();
            const unsigned char* st = ring + (s % S) * STAGE_BYTES;
#pragma unroll
            for (int j = 0; j < 4; ++j) {                         // k16 steps of the chunk, in order
                uint32_t b[4];
                const unsigned sa = (unsigned)__cvta_generic_to_shared(st + swz(lr, 2 * j + lc));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                             : "r"(sa));
                const int k = s * CHUNK + 16 * j;
#pragma unroll
                for (int r = 0; r < MT; ++r) {
                    const uint32_t a[4] = {x_pair(xr[r][0] + k, ok[r][0]), x_pair(xr[r][1] + k, ok[r][1]),
                                           x_pair(xr[r][0] + k + 8, ok[r][0]), x_pair(xr[r][1] + k + 8, ok[r][1])};
                    mma16816(acc[r][0], a, b[0], b[1]);
                    mma16816(acc[r][1], a, b[2], b[3]);
                }
            }
            __syncwarp();                                         // the stage is overwritten S - 1 steps later
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncwarp();
#pragma unroll
        for (int r = 0; r < MT; ++r)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int hi = 0; hi < 2; ++hi) {
                    const int m = m0 + 16 * r + g + 8 * hi, n = n0 + 8 * h + 2 * t;
                    if (m >= M) continue;
                    const float v0 = acc[r][h][2 * hi], v1 = acc[r][h][2 * hi + 1];
                    if (F32) {
                        float* o = static_cast<float*>(OUT) + ((size_t)slice * M + m) * N + n;
                        if (n + 1 < N) *reinterpret_cast<float2*>(o) = make_float2(v0, v1);
                        else if (n < N) o[0] = v0;
                    } else {
                        __nv_bfloat16* o = static_cast<__nv_bfloat16*>(OUT) + (size_t)m * o_stride + n;
                        if (n + 1 < N) *reinterpret_cast<__nv_bfloat162*>(o) = __floats2bfloat162_rn(v0, v1);
                        else if (n < N) o[0] = __float2bfloat16_rn(v0);
                    }
                }
    }
}


// The same arithmetic from the lossless 12-bit store (hc_pack in exl3_mm.py): a 64-k stage of 16 rows is 128 groups
// of 8 values in 3 words each (value i: bits 12 i.. of the 96: sign, fp16 exponent field 0-15, fp16 mantissa bits 9-3),
// decoded into an fp16 stage in shared memory, the stage's exceptions (other values, exact fp16) written over it, then
// read as hc_gemv_kernel reads its stage: the same fp16 operands, so the same bits.
constexpr int PSTAGE = 16 * CHUNK * 12 / 8;       // 1536 bytes

__device__ __forceinline__ uint32_t h12(uint32_t c) {     // 12-bit code -> fp16 bits
    return ((c & 0x800u) << 4) | ((c & 0x780u) << 3) | ((c & 0x7fu) << 3);
}

template <int MT, bool F32, int NW, int S>
__global__ void __launch_bounds__(NW * 32) hc_gemv12_kernel(
        const __nv_bfloat16* __restrict__ X, const unsigned char* __restrict__ P, const int* __restrict__ eoff,
        const uint32_t* __restrict__ ex, void* __restrict__ OUT, int M, int N, int K, int KS, int x_stride,
        int o_stride, int tiles, int items) {
    __shared__ __align__(128) unsigned char ring_all[NW][S][PSTAGE];
    __shared__ __align__(128) unsigned char half_all[NW][STAGE_BYTES];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int item = blockIdx.x * NW + warp;
    if (item >= items) return;
    const int tile = item % tiles, slice = item / tiles;
    const int k0 = slice * KS, steps = KS / CHUNK, KC = K / CHUNK;
    const int blk0 = tile * KC + slice * steps;                  // this warp's first stage block
    const int g = lane >> 2, t = lane & 3;
    unsigned char* ring = &ring_all[warp][0][0];
    unsigned char* hs = &half_all[warp][0];
    const int n0 = tile * 16;
    // exception offsets of the warp's stages (steps <= 63): lane j holds stage j's and stage j + 32's
    const int off_lo = __ldg(eoff + blk0 + min(lane, steps)), off_hi = __ldg(eoff + blk0 + min(lane + 32, steps));
    auto off_of = [&](int j) {
        const int a = __shfl_sync(0xffffffffu, off_lo, j & 31), b = __shfl_sync(0xffffffffu, off_hi, j & 31);
        return j < 32 ? a : b;
    };
    auto issue = [&](int s) {
        if (s < steps) {
            const unsigned char* src = P + (size_t)(blk0 + s) * PSTAGE;
            unsigned char* st = ring + (s % S) * PSTAGE;
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                const unsigned sa = (unsigned)__cvta_generic_to_shared(st + (lane + 32 * q) * 16);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(src + (lane + 32 * q) * 16));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    const int lr = (lane & 7) + ((lane >> 4) << 3), lc = (lane >> 3) & 1;

    for (int m0 = 0; m0 < M; m0 += 16 * MT) {
        float acc[MT][2][4];
#pragma unroll
        for (int r = 0; r < MT; ++r)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[r][h][c] = 0.f;
        const __nv_bfloat16* xr[MT][2];
        bool ok[MT][2];
#pragma unroll
        for (int r = 0; r < MT; ++r) {
            const int a0 = m0 + 16 * r + g, a1 = a0 + 8;
            ok[r][0] = a0 < M;
            ok[r][1] = a1 < M;
            xr[r][0] = X + (size_t)(ok[r][0] ? a0 : 0) * x_stride + k0 + 2 * t;
            xr[r][1] = X + (size_t)(ok[r][1] ? a1 : 0) * x_stride + k0 + 2 * t;
        }
#pragma unroll
        for (int s = 0; s < S - 1; ++s) issue(s);
        int e_a = off_of(0), e_b = off_of(1);
        uint32_t e_next = (lane < e_b - e_a) ? __ldg(ex + e_a + lane) : 0u;
#pragma unroll 1
        for (int s = 0; s < steps; ++s) {
            issue(s + S - 1);
            const uint32_t e_cur = e_next;
            const int cnt = e_b - e_a, e_base = e_a;
            if (s + 1 < steps) {                                  // the next stage's exceptions, one stage ahead
                e_a = e_b;
                e_b = off_of(s + 2);
                e_next = (lane < e_b - e_a) ? __ldg(ex + e_a + lane) : 0u;
            }
            asm volatile("cp.async.wait_group %0;\n" ::"n"(S - 1));
            __syncwarp();
            const unsigned char* st = ring + (s % S) * PSTAGE;
#pragma unroll
            for (int i = 0; i < 4; ++i) {                         // group q = 8 values: row q >> 3, 16-byte chunk q & 7
                const int q = lane + 32 * i;
                const uint32_t* w = reinterpret_cast<const uint32_t*>(st + q * 12);
                const uint32_t w0 = w[0], w1 = w[1], w2 = w[2];
                const uint32_t c0 = w0 & 0xfffu, c1 = (w0 >> 12) & 0xfffu, c2 = ((w0 >> 24) | (w1 << 8)) & 0xfffu,
                               c3 = (w1 >> 4) & 0xfffu, c4 = (w1 >> 16) & 0xfffu,
                               c5 = ((w1 >> 28) | (w2 << 4)) & 0xfffu, c6 = (w2 >> 8) & 0xfffu, c7 = w2 >> 20;
                uint4 v;
                v.x = h12(c0) | (h12(c1) << 16);
                v.y = h12(c2) | (h12(c3) << 16);
                v.z = h12(c4) | (h12(c5) << 16);
                v.w = h12(c6) | (h12(c7) << 16);
                *reinterpret_cast<uint4*>(hs + swz(q >> 3, q & 7)) = v;
            }
            __syncwarp();
            if (lane < cnt) {                                     // entry: (row * 64 + k) << 16 | fp16 bits
                const int at = e_cur >> 16, r = at >> 6, k = at & 63;
                *reinterpret_cast<unsigned short*>(hs + swz(r, k >> 3) + (k & 7) * 2) = (unsigned short)(e_cur & 0xffffu);
            }
            for (int e = 32 + lane; e < cnt; e += 32) {           // rare: more than 32 in a stage
                const uint32_t en = __ldg(ex + e_base + e);
                const int at = en >> 16, r = at >> 6, k = at & 63;
                *reinterpret_cast<unsigned short*>(hs + swz(r, k >> 3) + (k & 7) * 2) = (unsigned short)(en & 0xffffu);
            }
            __syncwarp();
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t b[4];
                const unsigned sa = (unsigned)__cvta_generic_to_shared(hs + swz(lr, 2 * j + lc));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                             : "r"(sa));
                const int k = s * CHUNK + 16 * j;
#pragma unroll
                for (int r = 0; r < MT; ++r) {
                    const uint32_t a[4] = {x_pair(xr[r][0] + k, ok[r][0]), x_pair(xr[r][1] + k, ok[r][1]),
                                           x_pair(xr[r][0] + k + 8, ok[r][0]), x_pair(xr[r][1] + k + 8, ok[r][1])};
                    mma16816(acc[r][0], a, b[0], b[1]);
                    mma16816(acc[r][1], a, b[2], b[3]);
                }
            }
            __syncwarp();
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncwarp();
#pragma unroll
        for (int r = 0; r < MT; ++r)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int hi = 0; hi < 2; ++hi) {
                    const int m = m0 + 16 * r + g + 8 * hi, n = n0 + 8 * h + 2 * t;
                    if (m >= M) continue;
                    const float v0 = acc[r][h][2 * hi], v1 = acc[r][h][2 * hi + 1];
                    if (F32) {
                        float* o = static_cast<float*>(OUT) + ((size_t)slice * M + m) * N + n;
                        if (n + 1 < N) *reinterpret_cast<float2*>(o) = make_float2(v0, v1);
                        else if (n < N) o[0] = v0;
                    } else {
                        __nv_bfloat16* o = static_cast<__nv_bfloat16*>(OUT) + (size_t)m * o_stride + n;
                        if (n + 1 < N) *reinterpret_cast<__nv_bfloat162*>(o) = __floats2bfloat162_rn(v0, v1);
                        else if (n < N) o[0] = __float2bfloat16_rn(v0);
                    }
                }
    }
}
}  // namespace

// out: fp32 [SK, M, N] (f32) or bf16 [M, >= N] rows (o_stride); x bf16 [M, K] (x_stride), w fp16 [N, K]
void hc_gemv(const at::Tensor& x, const at::Tensor& w, at::Tensor out, int64_t sk, bool f32) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1, "x: bf16 rows");
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kHalf && w.is_contiguous(), "w: contiguous fp16 [N, K]");
    const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
    TORCH_CHECK(w.size(1) == K && K % sk == 0 && (K / sk) % CHUNK == 0, "K slices must be multiples of 64");
    TORCH_CHECK(x.stride(0) % 2 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 4 == 0, "x rows 4-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 && K % 8 == 0, "w 16-byte aligned rows");
    if (f32) {
        TORCH_CHECK(out.scalar_type() == at::kFloat && out.is_contiguous() && out.numel() >= sk * M * N, "out: fp32 [SK, M, N]");
    } else {
        TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.stride(1) == 1 && sk == 1 && N % 2 == 0, "out: bf16 rows");
    }
    c10::cuda::CUDAGuard guard(x.device());
    const int tiles = (N + 15) / 16, items = tiles * (int)sk;
    const dim3 grid((items + WARPS - 1) / WARPS);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int mt = M <= 16 ? 1 : M <= 32 ? 2 : 4;
#define L(MT_, F_)                                                                                               \
    hc_gemv_kernel<MT_, F_><<<grid, WARPS * 32, 0, stream>>>(                                                    \
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const half*>(w.data_ptr()),        \
        out.data_ptr(), M, N, K, K / (int)sk, (int)x.stride(0), f32 ? N : (int)out.stride(0), tiles, items)
    if (f32) { if (mt == 1) L(1, true); else if (mt == 2) L(2, true); else L(4, true); }
    else { if (mt == 1) L(1, false); else if (mt == 2) L(2, false); else L(4, false); }
#undef L
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the 12-bit store: packed uint8 [tiles * K/64 * 1536], eoff int32 [tiles * K/64 + 1], ex int32 entries; N, K given
void hc_gemv12(const at::Tensor& x, const at::Tensor& packed, const at::Tensor& eoff, const at::Tensor& ex, int64_t N,
               at::Tensor out, int64_t sk, bool f32, int64_t cfg) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1, "x: bf16 rows");
    const int M = (int)x.size(0), K = (int)x.size(1);
    TORCH_CHECK(K % sk == 0 && (K / sk) % CHUNK == 0 && K / sk / CHUNK <= 63, "K slices: multiples of 64, <= 63 stages");
    const int tiles = (int)(N + 15) / 16, items = tiles * (int)sk;
    TORCH_CHECK(packed.numel() == (int64_t)tiles * (K / CHUNK) * PSTAGE && eoff.numel() == (int64_t)tiles * (K / CHUNK) + 1,
                "packed store does not match N, K");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(packed.data_ptr()) % 16 == 0, "packed: 16-byte aligned");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int mt = M <= 16 ? 1 : M <= 32 ? 2 : 4;
    // launch shape (warps a block, stages a warp): no effect on any output's arithmetic
    const int nw = cfg == 1 || cfg == 3 ? 2 : 4, st = cfg >= 2 ? 3 : 4;
    const dim3 grid((items + nw - 1) / nw);
#define L(MT_, F_, NW_, S_)                                                                                      \
    hc_gemv12_kernel<MT_, F_, NW_, S_><<<grid, NW_ * 32, 0, stream>>>(                                           \
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), packed.data_ptr<uint8_t>(), eoff.data_ptr<int>(),   \
        reinterpret_cast<const uint32_t*>(ex.data_ptr<int>()), out.data_ptr(), M, (int)N, K, K / (int)sk,        \
        (int)x.stride(0), f32 ? (int)N : (int)out.stride(0), tiles, items)
#define LS(MT_, F_) { if (nw == 2 && st == 4) L(MT_, F_, 2, 4); else if (nw == 2) L(MT_, F_, 2, 3); \
                      else if (st == 4) L(MT_, F_, 4, 4); else L(MT_, F_, 4, 3); }
    if (f32) { if (mt == 1) LS(1, true) else if (mt == 2) LS(2, true) else LS(4, true) }
    else { if (mt == 1) LS(1, false) else if (mt == 2) LS(2, false) else LS(4, false) }
#undef LS
#undef L
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("hc_gemv", &hc_gemv); m.def("hc_gemv12", &hc_gemv12, py::arg("x"), py::arg("packed"), py::arg("eoff"), py::arg("ex"), py::arg("N"), py::arg("out"), py::arg("sk"), py::arg("f32"), py::arg("cfg") = 0); }
