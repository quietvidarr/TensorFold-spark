// Prompt-window EXL3 experts: one program per (expert, 16 * W * NTW columns, matrix) decodes each weight tile once
// and multiplies it into every member of the expert (MT member tiles a pass), the members' inputs staged in shared
// memory by cp.async. Each output sums the whole K in order in one warp, so a row never depends on the window or on
// the other members; Z gets one split (SK = 1) for the existing epilogues.
#pragma once

#include "experts_grouped.cuh"

namespace tf_exl3x {

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// A program: WM x WN warps; warp (wm, wn) owns MTW member tiles (rows) and NTW column tiles. Per stage the block
// stages BK of K for its ROWS members (inputs) and for its WN * NTW column tiles (trellis words); both warp rows share
// the staged words, so each weight is read from memory once a pass.
template <int WM, int WN, int MTW, int NTW, int BK, int AST, int HI, bool ROT>
struct PrefillLayout {
    static constexpr int ROWS = WM * MTW * 16, KTS = BK / 16, TWM = 4 * HI, CT = WN * NTW;
    // ROT: the members' rotated inputs of one 128-wide K chunk, plus the next chunk's raw rows and suh; else AST
    // stages of BK of the inputs
    static constexpr int LD = ROT ? 128 + 8 : BK + 8;
    static constexpr int A_STAGES = ROT ? 1 : AST;
    static constexpr int A_STAGE = ROWS * LD;                      // halves
    static constexpr int W_STAGE = KTS * CT * TWM;                 // words
    static constexpr size_t A_BYTES = (size_t)A_STAGES * A_STAGE * 2;
    static constexpr size_t RAW_BYTES = ROT ? (size_t)ROWS * 128 * 2 + 128 * 2 : 0;
    static constexpr size_t W_BYTES = (size_t)AST * W_STAGE * 4;
    static constexpr size_t BYTES = A_BYTES + RAW_BYTES + W_BYTES + ROWS * 4;
    __device__ static half* a(char* base, int st) { return reinterpret_cast<half*>(base) + (size_t)st * A_STAGE; }
    __device__ static __nv_bfloat16* raw(char* base) { return reinterpret_cast<__nv_bfloat16*>(base + A_BYTES); }
    __device__ static half* raw_suh(char* base) { return reinterpret_cast<half*>(base + A_BYTES + (size_t)ROWS * 256); }
    __device__ static uint32_t* w(char* base, int st) {
        return reinterpret_cast<uint32_t*>(base + A_BYTES + RAW_BYTES) + (size_t)st * W_STAGE;
    }
    __device__ static int* rows(char* base) { return reinterpret_cast<int*>(base + A_BYTES + RAW_BYTES + W_BYTES); }
};

// The Walsh-Hadamard transform of 128 values, 4 a lane: ``rot_in``'s butterfly, in its order (bit for bit).
__device__ __forceinline__ void pf_fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

// ROT: the members' raw rows [k0, k0 + 128) and suh's, into the raw buffer (absent members: zeros).
template <class L, int THREADS>
__device__ __forceinline__ void stage_raw(char* sm, const __nv_bfloat16* __restrict__ XR, int xr_stride,
                                          const half* __restrict__ suh, int slots, int k0) {
    const int* rows = L::rows(sm);
    __nv_bfloat16* raw = L::raw(sm);
    for (int i = threadIdx.x; i < L::ROWS * 16; i += THREADS) {
        const int row = i >> 4, piece = i & 15;
        const int p = rows[row];
        cp_async16(raw + row * 128 + piece * 8, XR + (size_t)(p < 0 ? 0 : p / slots) * xr_stride + k0 + piece * 8,
                   p >= 0);
    }
    if (threadIdx.x < 16) cp_async16(L::raw_suh(sm) + threadIdx.x * 8, suh + k0 + threadIdx.x * 8, true);
}

// rot_in's arithmetic on the staged chunk: fp16((x * suh) @ H / sqrt(128)), a row a warp at a time.
template <class L, int WARPS>
__device__ __forceinline__ void rotate_raw(char* sm, int warp, int lane) {
    const int* rows = L::rows(sm);
    const __nv_bfloat16* raw = L::raw(sm);
    half* A = L::a(sm, 0);
    const uint2 su = *reinterpret_cast<const uint2*>(L::raw_suh(sm) + 4 * lane);
    const half2 s01 = *reinterpret_cast<const half2*>(&su.x), s23 = *reinterpret_cast<const half2*>(&su.y);
    const float sv[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
    for (int row = warp; row < L::ROWS; row += WARPS) {
        const uint2 xv = *reinterpret_cast<const uint2*>(raw + row * 128 + 4 * lane);
        const __nv_bfloat162 x01 = *reinterpret_cast<const __nv_bfloat162*>(&xv.x);
        const __nv_bfloat162 x23 = *reinterpret_cast<const __nv_bfloat162*>(&xv.y);
        float v[4] = {__low2float(x01) * sv[0], __high2float(x01) * sv[1], __low2float(x23) * sv[2],
                      __high2float(x23) * sv[3]};
        pf_fwht128(v, lane);
        const bool ok = rows[row] >= 0;
        constexpr float S = 0.08838834764831845f;
        half2 o01 = __floats2half2_rn(ok ? v[0] * S : 0.f, ok ? v[1] * S : 0.f);
        half2 o23 = __floats2half2_rn(ok ? v[2] * S : 0.f, ok ? v[3] * S : 0.f);
        uint2 o;
        o.x = *reinterpret_cast<uint32_t*>(&o01);
        o.y = *reinterpret_cast<uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(A + row * L::LD + 4 * lane) = o;
    }
}

// mma16816 without ``volatile`` (a pure function of its registers), so the compiler may interleave the next tile's
// decode with these multiplies; the grouped decode kernel keeps its own.
__device__ __forceinline__ void mma16816_pure(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The m16n8k16 A fragment of a 16x16 fp16 tile (row stride ld halves) in one ldmatrix: the same values as four
// 32-bit loads of rows g, g + 8 at columns 2t and 2t + 8.
__device__ __forceinline__ void ldmatrix_a(uint32_t (&a)[4], const half* tile, int ld, int lane) {
    const unsigned addr = (unsigned)__cvta_generic_to_shared(tile + (lane & 15) * ld + (lane >> 4) * 8);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
}

template <class L, int K2, int THREADS, bool ROT>
__device__ __forceinline__ void stage(char* sm, int st, const half* __restrict__ X, int K, int k0,
                                      const uint32_t* __restrict__ T, size_t kstride, int ct0) {
    if constexpr (!ROT) {
        constexpr int CHUNKS = (L::LD - 8) / 8;        // 16-byte pieces of a staged input row
        const int* rows = L::rows(sm);
        half* a = L::a(sm, st);
        for (int i = threadIdx.x; i < L::ROWS * CHUNKS; i += THREADS) {
            const int row = i / CHUNKS, c = i % CHUNKS;
            const int r = rows[row];
            cp_async16(a + row * L::LD + c * 8, X + (size_t)(r < 0 ? 0 : r) * K + k0 + c * 8, r >= 0);
        }
    }
    constexpr int TW = Fmt<K2>::TW, PIECES = TW / 4;   // 16-byte pieces of a trellis tile
    uint32_t* w = L::w(sm, st);
    const int kt = k0 >> 4;
    for (int i = threadIdx.x; i < L::KTS * L::CT * PIECES; i += THREADS) {
        const int tile = i / PIECES, piece = i % PIECES;
        const int kk = tile / L::CT, n = tile % L::CT;
        cp_async16(w + (kk * L::CT + n) * L::TWM + piece * 4,
                   T + (size_t)(kt + kk) * kstride + (size_t)(ct0 + n) * TW + piece * 4, true);
    }
}

template <int CB, int K2, int WM, int WN, int MTW, int NTW, int BK, int AST, bool ROT, class L>
__device__ __forceinline__ void prefill_strip(char* sm, const half* __restrict__ X, const uint32_t* __restrict__ T,
                                              float* __restrict__ Zm, int K, int N, int ct0,
                                              const __nv_bfloat16* __restrict__ XR, int xr_stride,
                                              const half* __restrict__ suh, int slots) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW, KTS = BK / 16, THREADS = WM * WN * 32;
    constexpr int WARPS = WM * WN, KCH = 128 / BK;      // ROT: K stages a 128-wide chunk
    static_assert(!ROT || 128 % BK == 0, "a rotation chunk is whole stages");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / WN, wn = warp % WN;
    const int NTILES = N >> 4, KT = K >> 4;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const int* rows = L::rows(sm);
    int active = 0;                                     // this warp's member tiles holding a member
#pragma unroll
    for (int mt = 0; mt < MTW; ++mt) active += rows[(wm * MTW + mt) * 16] >= 0;

    float acc[MTW][NTW][2][4];
#pragma unroll
    for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
        for (int i = 0; i < NTW; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[mt][i][h][c] = 0.f;

    const int stages = KT / KTS;
    if constexpr (ROT) stage_raw<L, THREADS>(sm, XR, xr_stride, suh, slots, 0);   // committed with stage 0
#pragma unroll
    for (int p = 0; p < AST - 1; ++p) {                 // AST - 1 stages in flight ahead
        if (p < stages) stage<L, K2, THREADS, ROT>(sm, p, X, K, p * BK, T, kstride, ct0);
        cp_async_commit();
    }
    for (int s = 0; s < stages; ++s) {
        const int b = s % AST;
        cp_async_wait<AST - 2>();                       // stage s has landed (and chunk s / KCH's raw rows)
        __syncthreads();                                // ... for every warp, and all are done with stage s - 1
        if constexpr (ROT) {
            if (s % KCH == 0) {                         // a new chunk: rotate it, then stage the next one's raw rows
                rotate_raw<L, WARPS>(sm, warp, lane);
                __syncthreads();
                if (s + KCH < stages) stage_raw<L, THREADS>(sm, XR, xr_stride, suh, slots, (s + KCH) * BK);
            }
        }
        if (s + AST - 1 < stages) stage<L, K2, THREADS, ROT>(sm, (s + AST - 1) % AST, X, K, (s + AST - 1) * BK, T, kstride, ct0);
        cp_async_commit();
        if (active == 0) continue;
        const half* A = ROT ? L::a(sm, 0) + (wm * MTW * 16) * L::LD + (s % KCH) * BK
                            : L::a(sm, b) + (wm * MTW * 16) * L::LD;
        const uint32_t* WS = L::w(sm, b);
#pragma unroll
        for (int kk = 0; kk < KTS; ++kk) {
            uint32_t bf[NTW][2][2];                     // every column tile's B fragments first, then the multiplies
#pragma unroll
            for (int i = 0; i < NTW; ++i) {
                uint32_t w[LW];
                const uint32_t* tile = WS + (kk * L::CT + wn * NTW + i) * L::TWM;
#pragma unroll
                for (int l = 0; l < LW; ++l) w[l] = (l * 32 + lane < TW) ? tile[l * 32 + lane] : 0u;
                decode_tile<CB, K2>(w, map, lane, bf[i][0], bf[i][1]);
            }
            if (active == MTW) {                        // a full pass (most): every fragment loaded first, no branch
                uint32_t a[MTW][4];
#pragma unroll
                for (int mt = 0; mt < MTW; ++mt) ldmatrix_a(a[mt], A + mt * 16 * L::LD + kk * 16, L::LD, lane);
#pragma unroll
                for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
                    for (int i = 0; i < NTW; ++i) {
                        mma16816_pure(acc[mt][i][0], a[mt], bf[i][0]);
                        mma16816_pure(acc[mt][i][1], a[mt], bf[i][1]);
                    }
                continue;
            }
#pragma unroll
            for (int mt = 0; mt < MTW; ++mt) {
                if (mt >= active) break;
                uint32_t a[4];
                ldmatrix_a(a, A + mt * 16 * L::LD + kk * 16, L::LD, lane);
#pragma unroll
                for (int i = 0; i < NTW; ++i) {
                    mma16816_pure(acc[mt][i][0], a, bf[i][0]);
                    mma16816_pure(acc[mt][i][1], a, bf[i][1]);
                }
            }
        }
    }
    cp_async_wait<0>();
#pragma unroll
    for (int mt = 0; mt < MTW; ++mt) {
        if (mt >= active) break;
        const int r0 = rows[(wm * MTW + mt) * 16 + g], r1 = rows[(wm * MTW + mt) * 16 + g + 8];
#pragma unroll
        for (int i = 0; i < NTW; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = (ct0 + wn * NTW + i) * 16 + h * 8 + 2 * t;
                if (r0 >= 0) *reinterpret_cast<float2*>(Zm + (size_t)r0 * N + col) = make_float2(acc[mt][i][h][0], acc[mt][i][h][1]);
                if (r1 >= 0) *reinterpret_cast<float2*>(Zm + (size_t)r1 * N + col) = make_float2(acc[mt][i][h][2], acc[mt][i][h][3]);
            }
    }
}

// grid (work item, column strip of WN * NTW tiles, matrix): a work item is one pass of ROWS members of one expert,
// from ``group_any``'s work list (work[0] items; work[1 + u] = expert u's first), so an expert every row picks (the
// shared one) spreads over many programs.
// ROT: the inputs are the raw rows XR [rows, K] bf16, rotated in the kernel with suh (rot_in's arithmetic), instead
// of X0 / X1 [P, K] fp16 written by rot_in for every slot.
template <int CB, int LO, int HI, int WM, int WN, int MTW, int NTW, int BK, int AST, bool ROT>
__global__ void __launch_bounds__(WM * WN * 32, 1) prefill_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    const int* __restrict__ work, float* __restrict__ Z, int K, int N, int P, int maxm, int slots,
    const __nv_bfloat16* __restrict__ XR, int xr_stride, const half* __restrict__ SUH0, const half* __restrict__ SUH1,
    int skip_k2) {
    using L = PrefillLayout<WM, WN, MTW, NTW, BK, AST, HI, ROT>;
    constexpr int THREADS = WM * WN * 32;
    const int item = blockIdx.y;                        // a work item's column strips are neighbours (blockIdx.x),
    if (item >= work[0]) return;                        // so they run together and share its inputs in L2
    int lo = 0, hi = ucount[0] - 1;                     // the last expert whose first item is <= item
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (work[1 + mid] <= item) lo = mid; else hi = mid - 1;
    }
    const int u = lo, pass = item - work[1 + u];
    const int mat = blockIdx.z;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    if (k2 < LO || k2 > HI || k2 == skip_k2) return;    // another launch's width (see prefill_launch)
    const int ct0 = blockIdx.x * L::CT;
    float* Zm = Z + (size_t)mat * P * N;
    extern __shared__ __align__(16) char sm[];
    int* rows = L::rows(sm);
    for (int i = threadIdx.x; i < L::ROWS; i += THREADS) {
        const int m = pass * L::ROWS + i;
        const int code = m < maxm ? members[(size_t)u * maxm + m] : -1;
        rows[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows[0] < 0) return;
    switch (k2) {
#define TF_PF_CASE(K2_)                                                                                         \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            prefill_strip<CB, K2_, WM, WN, MTW, NTW, BK, AST, ROT, L>(sm, X, T, Zm, K, N, ct0, XR, xr_stride,  \
                                                                       (mat ? SUH1 : SUH0) + (size_t)e * K, slots); \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_PF_CASE(2) TF_PF_CASE(3) TF_PF_CASE(4) TF_PF_CASE(5) TF_PF_CASE(6) TF_PF_CASE(7) TF_PF_CASE(8)
        TF_PF_CASE(9) TF_PF_CASE(10) TF_PF_CASE(11) TF_PF_CASE(12) TF_PF_CASE(13) TF_PF_CASE(14)
        TF_PF_CASE(15) TF_PF_CASE(16)
#undef TF_PF_CASE
        default:
            __trap();
    }
}

struct PrefillArgs {
    const half* x0;
    const half* x1;
    const int64_t* tp0;
    const int64_t* tp1;
    const int* k2_0;
    const int* k2_1;
    const int* uids;
    const int* ucount;
    const int* members;
    const int* work;
    float* z;
    int K, N, P, maxm, slots, nexp_max, mats, lo, hi;   // nexp_max: most work items
    const __nv_bfloat16* xr = nullptr;                  // set: rotate the raw rows in the kernel
    int xr_stride = 0;
    const half* suh0 = nullptr;
    const half* suh1 = nullptr;
};

constexpr int PF_COLS = 320;
constexpr int PF_ROWS = 128;                       // members a work item
constexpr int PF_BKMAX = 128;                      // K must split in it (a rotation chunk)

template <int CB, int LO, int HI, int WM, int WN, int MTW, int NTW, int BK, int AST, bool ROT>
void prefill_go_(const PrefillArgs& a, cudaStream_t stream, int skip_k2) {
    static_assert(WN * NTW * 16 == PF_COLS, "a program covers PF_COLS columns");
    using L = PrefillLayout<WM, WN, MTW, NTW, BK, AST, HI, ROT>;
    auto kern = prefill_kernel<CB, LO, HI, WM, WN, MTW, NTW, BK, AST, ROT>;
    static bool once = [&] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L::BYTES));
        return true;
    }();
    (void)once;
    dim3 grid((unsigned)(a.N / PF_COLS), (unsigned)a.nexp_max, (unsigned)a.mats);
    kern<<<grid, WM * WN * 32, L::BYTES, stream>>>(a.x0, a.x1, a.tp0, a.tp1, a.k2_0, a.k2_1, a.uids, a.ucount,
                                                    a.members, a.work, a.z, a.K, a.N, a.P, a.maxm, a.slots, a.xr,
                                                    a.xr_stride, a.suh0, a.suh1, skip_k2);
}

template <int CB, int LO, int HI, int WM, int WN, int MTW, int NTW, int BK, int AST>
void prefill_go(const PrefillArgs& a, cudaStream_t stream, int skip_k2 = -1) {
    if (a.xr) prefill_go_<CB, LO, HI, WM, WN, MTW, NTW, BK, AST, true>(a, stream, skip_k2);
    else prefill_go_<CB, LO, HI, WM, WN, MTW, NTW, BK, AST, false>(a, stream, skip_k2);
}

// Shared memory is sized for the widest trellis a launch takes. 4-bit experts (K2 8) get their own launch, sized for
// them, so two programs fit an SM; any other width (Flash Next's 6-bit shared expert) goes to a second, generic one.
template <int CB>
void prefill_launch(const PrefillArgs& a, cudaStream_t stream) {
    constexpr int W = 1, N = 10, M = 8, T = 2, B = 64, S = 2;  // 10 warps, 128 members, 320 columns, K 64 a stage, 2 deep
    if (a.lo <= 8 && a.hi >= 8) prefill_go<CB, 8, 8, W, N, M, T, B, S>(a, stream);
    if (a.lo == 8 && a.hi == 8) return;
    if (a.hi <= 12) prefill_go<CB, 2, 12, W, N, M, T, B, S>(a, stream, 8);   // e.g. a 6-bit shared expert
    else prefill_go<CB, 2, 16, W, N, M, T, 32, 2>(a, stream, 8);           // up to 8 bits: K 32 a stage fits
}

}  // namespace tf_exl3x
