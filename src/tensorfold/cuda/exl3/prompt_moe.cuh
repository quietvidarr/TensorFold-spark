// Prompt-window EXL3 MoE, second generation: two fused launches a layer instead of four kernels and two fp32
// partial buffers (lab patch; see experts.py _routed_prompt2).
//
//   gate/up (gu_kernel): a program = (work item = <= 128 members of one expert, which 64 of them, 128 columns of
//     the expert width), gate and up together; 4 warps, two programs an SM. The members' fp16 inputs (x * 2^-2, see
//     to_half) are staged 128 of K at a time and rotated on the tensor cores: (x * suh) @ H128 is, per k16 slice,
//     x @ (diag(suh) H16) (one mma; fp16 products are exact) and H8 across the 8 slices in fp32 registers; the
//     rotated rows stay in registers as A fragments. Each k16 step's 16 trellis tiles are decoded once a program
//     into shared memory as fp16 B fragments that all its warps read. Warp (wm, mat): 32 rows x 128 columns of gate
//     or up. Epilogue in registers: H128 over the warp's 128 columns (two lane-shuffle stages + register
//     butterflies), * svh, SwiGLU, * suh_d, H128 again -> fp16 xd, stored per work item in k16 slices of its 128
//     rows ([item][I / 16][128][16]) so down reads a slice as one 4 KB run; up's values reach gate's warp through
//     shared memory.
//   down (dn_kernel): a program = (work item, 128 model columns), 8 warps of 16 rows, two programs an SM; xd slices
//     by cp.async, the same decode; epilogue H128 * svh_d, then * wts and an fp32 red.add into out[row] (summed: out
//     zeroed first), or y [P, D]. Column groups are the grid's outermost dimension, so the programs running at a
//     time add into a slice of out that stays in L2.
//   The trellis pipelines are as deep as two programs an SM allow for the launch's widest trellis.
//
// Sums are fp32 (tensor-core accumulation) in an order that differs from the first-generation kernel's; with
// red.add (TF_EXL3_PROMPT_DET=0) the summed output's slot order varies from run to run; the default (DET=1) writes y
// and sums the slots in a fixed order.
#pragma once

#include "experts_grouped.cuh"

namespace tf_exl3p {

using tf_exl3x::Fmt;
using tf_exl3x::LaneMap;

constexpr int PASS = 128;                    // members a work item (experts.py PREFILL_PASS)
constexpr float HS = 0.08838834764831845f;   // 1 / sqrt(128)
constexpr float X_SCALE = 0.25f;             // x is staged as fp16(x * 2^-2): bf16's range in fp16
constexpr float GU_SCALE = 0.03125f;         // 2^-5: undoes 2^-2 and two unnormalized H128 (2^3.5 each)
constexpr int SMEM_PAIR = 50176;             // shared memory a program when two share an SM (2 * (this + 1 KB))

__device__ __forceinline__ void cp16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait_cp() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm4(uint32_t (&a)[4], const void* p) {
    const unsigned addr = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
}

// mma16816 without ``volatile`` (a pure function of its registers), so the compiler may interleave it
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ float2 h2f(uint32_t u) { return __half22float2(*reinterpret_cast<const half2*>(&u)); }

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// The unnormalized Walsh-Hadamard transform over a warp's 128 accumulator columns of one m16 tile: column
// 8 jj + 2 t + b (jj the n8 tile, t = lane & 3, b the register's low bit); rows g and g + 8 stay apart.
__device__ __forceinline__ void wht128(float (&v)[16][4], int lane) {
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        float a = v[jj][0], b = v[jj][1];
        v[jj][0] = a + b; v[jj][1] = a - b;
        a = v[jj][2]; b = v[jj][3];
        v[jj][2] = a + b; v[jj][3] = a - b;
    }
#pragma unroll
    for (int s = 1; s <= 2; s <<= 1) {
        const float sg = (lane & s) ? -1.f : 1.f;
#pragma unroll
        for (int jj = 0; jj < 16; ++jj)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float o = __shfl_xor_sync(0xffffffffu, v[jj][r], s);
                v[jj][r] = fmaf(sg, v[jj][r], o);          // bit clear: v + o; set: o - v
            }
    }
#pragma unroll
    for (int st = 1; st < 16; st <<= 1)
#pragma unroll
        for (int jj = 0; jj < 16; ++jj)
            if (!(jj & st))
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const float a = v[jj][r], b = v[jj + st][r];
                    v[jj][r] = a + b;
                    v[jj + st][r] = a - b;
                }
}

// The work item's expert (binary search of group_any's work list) and pass.
__device__ __forceinline__ bool find_item(const int* __restrict__ work, const int* __restrict__ ucount, int item,
                                          int& u, int& pass) {
    if (item >= work[0]) return false;
    int lo = 0, hi = ucount[0] - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (work[1 + mid] <= item) lo = mid; else hi = mid - 1;
    }
    u = lo;
    pass = item - work[1 + u];
    return true;
}

// items[i] = work item i's (expert slot, pass), (-1, 0) past the last: the programs read it instead of searching
__global__ void item_table_kernel(const int* __restrict__ work, const int* __restrict__ ucount, int2* __restrict__ items,
                                  int n) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= n) return;
    int u, pass;
    items[item] = find_item(work, ucount, item, u, pass) ? make_int2(u, pass) : make_int2(-1, 0);
}

// One k16 step's NT trellis tiles, this thread's pieces' addresses computed once a program. TWO: tiles nt < NT / 2
// of T0, the rest of T1, at column tiles ct0 + nt % (NT / 2); else T0 at ct0 + nt.
template <int K2, int NT, bool TWO, int NTHR>
struct WCopy {
    static constexpr int TW = 4 * K2, H = NT / 2, PIECES = NT * K2, N = (PIECES + NTHR - 1) / NTHR;
    const uint32_t* src[N];
    int dst[N];
    __device__ __forceinline__ WCopy(const uint32_t* T0, const uint32_t* T1, int ct0) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
            const int i = min((int)threadIdx.x + j * NTHR, PIECES - 1);
            const int nt = i / K2, pc = i % K2;
            const uint32_t* T = (TWO && nt >= H) ? T1 : T0;
            src[j] = T + (size_t)(ct0 + (TWO ? nt % H : nt)) * TW + pc * 4;
            dst[j] = nt * TW + pc * 4;
        }
    }
    __device__ __forceinline__ void operator()(uint32_t* ws, size_t off) const {
#pragma unroll
        for (int j = 0; j < N; ++j)
            if (PIECES % NTHR == 0 || (int)threadIdx.x + j * NTHR < PIECES) cp16(ws + dst[j], src[j] + off, true);
    }
};

// decode_tile (experts_grouped.cuh) with each lane reading the words it needs straight from shared memory
template <int CB, int K2>
__device__ __forceinline__ void decode_smem(const uint32_t* tile, const LaneMap<K2>& m, int lane, uint32_t (&b0)[2],
                                            uint32_t (&b1)[2]) {
    uint32_t st[8];
    if constexpr (K2 == 8) {
        const uint32_t w0 = tile[lane], p = tile[(lane + 31) & 31];
        const uint32_t s = __funnelshift_r(w0, p, 20);
        st[0] = (s >> 8) & 0xffffu;
        st[1] = (s >> 4) & 0xffffu;
        st[2] = s & 0xffffu;
        st[3] = w0 >> 16;
        st[4] = (w0 >> 12) & 0xffffu;
        st[5] = (w0 >> 8) & 0xffffu;
        st[6] = (w0 >> 4) & 0xffffu;
        st[7] = w0 & 0xffffu;
    } else {
        constexpr int GV = Fmt<K2>::GV, NG = Fmt<K2>::NG;
        uint32_t whi[NG], wlo[NG];
#pragma unroll
        for (int g = 0; g < NG; ++g) { whi[g] = tile[m.hi[g]]; wlo[g] = tile[m.lo[g]]; }
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const uint64_t mm = ((((uint64_t)wlo[g]) << 32) | whi[g]) >> m.sh[g];
#pragma unroll
            for (int j = 0; j < GV; ++j) st[g * GV + j] = (uint32_t)(mm >> Fmt<K2>::off(j)) & 0xffffu;
        }
    }
    b0[0] = tf_exl3x::cb_pair<CB>(st[0], st[1]);
    b0[1] = tf_exl3x::cb_pair<CB>(st[2], st[3]);
    b1[0] = tf_exl3x::cb_pair<CB>(st[4], st[5]);
    b1[1] = tf_exl3x::cb_pair<CB>(st[6], st[7]);
}

// tile nt of a stage -> bd[nt]: lane l's uint4 {b0 (n 0-7), b1 (n 8-15)}
template <int CB, int K2>
__device__ __forceinline__ void decode_one(const uint32_t* ws, uint4* bd, int nt, int lane, const LaneMap<K2>& map) {
    uint32_t b0[2], b1[2];
    decode_smem<CB, K2>(ws + nt * 4 * K2, map, lane, b0, b1);
    bd[nt * 32 + lane] = make_uint4(b0[0], b0[1], b1[0], b1[1]);
}

// ------------------------------------------------------------------------------------------------ gate / up

namespace gu {
constexpr int WARPS = 4, NTHR = WARPS * 32, BM = 64;   // 64 of a work item's members, two programs an SM
constexpr int NT = 16;                      // tiles a k16 step (8 gate + 8 up)
constexpr int ROWS = 2 * BM * 4;            // pair index and row of the program's 64 members
constexpr int SUH = 512;                    // gate's and up's suh of the chunk
constexpr int BD = 2 * NT * 512;            // B fragments, double-buffered
constexpr int XS = BM * 256;                // a 128-wide chunk of the rows' inputs
__host__ __device__ constexpr int wst(int k2) { return NT * 4 * k2; }   // words a trellis stage
// [rows][suh][B fragments][x chunk][trellis stages]: B fragments + x chunk (32 KB) hold up's values for the epilogue
__host__ __device__ constexpr size_t smem(int k2, int ns) { return ROWS + SUH + BD + XS + (size_t)ns * wst(k2) * 4; }
// the deepest trellis pipeline two programs an SM fit for trellises up to k2 (8: 8 stages, 12: 5, 16: 4)
__host__ __device__ constexpr int depth(int k2) { return (int)((SMEM_PAIR - ROWS - SUH - BD - XS) / (wst(k2) * 4)); }
}  // namespace gu

struct GuArgs {
    const half* xh;                 // [R, D] fp16(x * 2^-2)
    const int64_t* tp0;             // gate trellis pointers [E]
    const int64_t* tp1;             // up
    const int* k2;                  // [E] (gate's = up's)
    const half* suh0;               // [E, D]
    const half* suh1;
    const half* svh0;               // [E, I]
    const half* svh1;
    const half* suh_d;              // [E, I]
    const int* uids;
    const int* ucount;
    const int* members;
    const int* work;
    const int2* items;              // [work items]: (expert slot u, pass), u -1 past the last
    half* xd;                       // [work items, I / 16, 128, 16]
    float* nrm;                     // optional [P, I / 128]: each (row, slot)'s sum of squares of its fp16 xd, a column group each
    int D, I, maxm, slots;
    float limit;
    int act_mode;
};

// The chunk's inputs (64 rows x 128 halves, 16-byte pieces swizzled by row & 7; absent rows zero) and gate's and
// up's suh.
__device__ __forceinline__ void stage_x(half* xs, half* ss, const half* __restrict__ xh, const int* xrow, int D, int k0,
                                        const half* __restrict__ suh0, const half* __restrict__ suh1) {
#pragma unroll
    for (int j = 0; j < gu::BM * 16 / gu::NTHR; ++j) {
        const int i = threadIdx.x + j * gu::NTHR;
        const int row = i >> 4, pc = i & 15;
        const int r = xrow[row];
        cp16(xs + row * 128 + ((pc ^ (row & 7)) << 3), xh + (size_t)(r < 0 ? 0 : r) * D + k0 + pc * 8, r >= 0);
    }
    if (threadIdx.x < 32) {
        const int m = threadIdx.x >> 4, pc = threadIdx.x & 15;
        cp16(ss + m * 128 + pc * 8, (m ? suh1 : suh0) + k0 + pc * 8, true);
    }
}

// Rows [r0, r0 + 16) of the chunk rotated: A[a] = the A fragment of k16 slice a of (x 2^-2 * suh) @ H128
// (unnormalized), fp16; one n8 half of the result columns at a time (32 fp32 temporaries). H128 = H8 (x) H16: the
// mma applies H16 within each slice with suh folded into its B operand (suh's sign bits flipped where H16 is -1),
// the butterflies H8 across the slices.
__device__ __forceinline__ void rotate_tile(uint32_t (&A)[8][4], const half* xs, int r0, const half* ss,
                                            const uint32_t (&mask)[2][2], int lane) {
    const int t = lane & 3;
    const int row = r0 + (lane & 15);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        float C[8][4];
#pragma unroll
        for (int a = 0; a < 8; ++a) {
            uint32_t xa[4];
            const int pc = 2 * a + (lane >> 4);
            ldsm4(xa, xs + row * 128 + ((pc ^ (row & 7)) << 3));
            const uint32_t s01 = *reinterpret_cast<const uint32_t*>(ss + 16 * a + 2 * t);
            const uint32_t s89 = *reinterpret_cast<const uint32_t*>(ss + 16 * a + 2 * t + 8);
#pragma unroll
            for (int r = 0; r < 4; ++r) C[a][r] = 0.f;
            mma(C[a], xa, s01 ^ mask[h][0], s89 ^ mask[h][1]);
        }
#pragma unroll
        for (int st = 1; st < 8; st <<= 1)
#pragma unroll
            for (int a = 0; a < 8; ++a)
                if (!(a & st))
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        const float p = C[a][r], q = C[a + st][r];
                        C[a][r] = p + q;
                        C[a + st][r] = p - q;
                    }
#pragma unroll
        for (int a = 0; a < 8; ++a) {
            A[a][2 * h] = pack_h2(C[a][0], C[a][1]);
            A[a][2 * h + 1] = pack_h2(C[a][2], C[a][3]);
        }
    }
}

// Warp (wm, mat): rows [32 wm, 32 wm + 32) of the program's 64, gate's or up's 128 columns.
template <int CB, int K2, int NS>
__device__ __forceinline__ void gu_body(char* sm, const GuArgs& g, int e, int half_) {
    constexpr int TW = 4 * K2, NT = gu::NT, WST = gu::wst(K2);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp & 1, mat = warp >> 1;
    const int t = lane & 3, gq = lane >> 2;
    const int D = g.D, I = g.I;
    const int* rows = reinterpret_cast<const int*>(sm);
    const int* xrow = rows + gu::BM;
    half* ss = reinterpret_cast<half*>(sm + gu::ROWS);
    uint4* bd = reinterpret_cast<uint4*>(sm + gu::ROWS + gu::SUH);
    half* xs = reinterpret_cast<half*>(sm + gu::ROWS + gu::SUH + gu::BD);
    uint32_t* wbuf = reinterpret_cast<uint32_t*>(sm + gu::ROWS + gu::SUH + gu::BD + gu::XS);
    const uint32_t* T0 = reinterpret_cast<const uint32_t*>(g.tp0[e]);
    const uint32_t* T1 = reinterpret_cast<const uint32_t*>(g.tp1[e]);
    const half* suh0 = g.suh0 + (size_t)e * D;
    const half* suh1 = g.suh1 + (size_t)e * D;
    const size_t kstride = (size_t)(I >> 4) * TW;
    const int grp = blockIdx.x >> 1, ct0 = grp * 8;
    const int KS = D / 16, chunks = D / 128;
    const LaneMap<K2> map(lane);

    // H16 signs of this lane's B entries: k in {2t, 2t+1} (reg 0) / {2t+8, 2t+9} (reg 1), n = 8h + g
    uint32_t mask[2][2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int n = 8 * h + gq;
        mask[h][0] = ((__popc((2 * t) & n) & 1) << 15) | ((__popc((2 * t + 1) & n) & 1) << 31);
        mask[h][1] = ((__popc((2 * t + 8) & n) & 1) << 15) | ((__popc((2 * t + 9) & n) & 1) << 31);
    }
    const int act = (rows[32 * wm] >= 0) + (rows[32 * wm + 16] >= 0);   // this warp's m16 tiles holding members

    float acc[2][16][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int j = 0; j < 16; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[m][j][r] = 0.f;
    uint32_t A[2][8][4];

    const WCopy<K2, NT, true, gu::NTHR> wcopy(T0, T1, ct0);
    stage_x(xs, ss, g.xh, xrow, D, 0, suh0, suh1);
#pragma unroll
    for (int p = 0; p < NS - 1; ++p) {
        wcopy(wbuf + p * WST, (size_t)p * kstride);
        commit();
    }
    wait_cp<NS - 2>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < NT / 4; ++i) decode_one<CB, K2>(wbuf, bd, warp + 4 * i, lane, map);

    for (int c = 0; c < chunks; ++c) {
#pragma unroll
        for (int sc = 0; sc < 8; ++sc) {
            const int s = 8 * c + sc;
            wait_cp<NS - 3>();                          // trellis stage s + 1 (and the chunk's inputs) landed
            __syncthreads();
            if (sc == 0) {
                if (act > 0) rotate_tile(A[0], xs, 32 * wm, ss + mat * 128, mask, lane);
                if (act > 1) rotate_tile(A[1], xs, 32 * wm + 16, ss + mat * 128, mask, lane);
                __syncthreads();                        // xs is free: the next chunk's inputs
                if (c + 1 < chunks) stage_x(xs, ss, g.xh, xrow, D, 128 * (c + 1), suh0, suh1);
            }
            if (s + NS - 1 < KS) wcopy(wbuf + ((s + NS - 1) % NS) * WST, (size_t)(s + NS - 1) * kstride);
            commit();
            if (s + 1 < KS) {
#pragma unroll
                for (int i = 0; i < NT / 4; ++i)
                    decode_one<CB, K2>(wbuf + ((s + 1) % NS) * WST, bd + ((s + 1) & 1) * NT * 32, warp + 4 * i, lane, map);
            }
            if (act == 0) continue;
            const uint4* bs = bd + (s & 1) * NT * 32 + mat * 8 * 32 + lane;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint4 b = bs[j * 32];
                mma(acc[0][2 * j], A[0][sc], b.x, b.y);
                mma(acc[0][2 * j + 1], A[0][sc], b.z, b.w);
                if (act > 1) {
                    mma(acc[1][2 * j], A[1][sc], b.x, b.y);
                    mma(acc[1][2 * j + 1], A[1][sc], b.z, b.w);
                }
            }
        }
    }
    wait_cp<0>();
    __syncthreads();                                    // B fragments + x chunk are free: up's values go through them

    float4* ex = reinterpret_cast<float4*>(sm + gu::ROWS + gu::SUH);   // [wm][m][jj][lane]: 32 KB
    const int col0 = grp * 128 + 2 * t;
    const float lim = g.limit;
    uint32_t sv[16], dv[16];
    {
        const half* svh = (mat ? g.svh1 : g.svh0) + (size_t)e * I + col0;
        const half* sd = g.suh_d + (size_t)e * I + col0;
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            sv[jj] = __ldg(reinterpret_cast<const unsigned int*>(svh + 8 * jj));
            dv[jj] = mat ? 0u : __ldg(reinterpret_cast<const unsigned int*>(sd + 8 * jj));
        }
    }
    if (mat == 1) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            if (m >= act) break;
            wht128(acc[m], lane);
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) {
                const float2 s = h2f(sv[jj]);
                float u[4];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    float v = __fmul_rn(__fmul_rn(acc[m][jj][r], GU_SCALE), (r & 1) ? s.y : s.x);
                    if (g.act_mode == 0) v = bf16r(v);
                    u[r] = fminf(fmaxf(v, -lim), lim);
                }
                ex[((wm * 2 + m) * 16 + jj) * 32 + lane] = make_float4(u[0], u[1], u[2], u[3]);
            }
        }
    }
    __syncthreads();
    if (mat == 1) return;
    half* xo = g.xd + (size_t)blockIdx.y * PASS * I + (size_t)(grp * 8) * PASS * 16 + 2 * t;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        if (m >= act) break;
        wht128(acc[m], lane);
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const float2 s = h2f(sv[jj]), d = h2f(dv[jj]);
            const float4 u4 = ex[((wm * 2 + m) * 16 + jj) * 32 + lane];
            const float u[4] = {u4.x, u4.y, u4.z, u4.w};
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                float gg = __fmul_rn(__fmul_rn(acc[m][jj][r], GU_SCALE), (r & 1) ? s.y : s.x);
                float a;
                if (g.act_mode == 0) {
                    gg = fminf(bf16r(gg), lim);
                    a = bf16r(bf16r(__fdividef(gg, 1.f + __expf(-gg))) * u[r]);
                } else {
                    gg = fminf(gg, lim);
                    a = __fmul_rn(__fdividef(gg, 1.f + __expf(-gg)), u[r]);
                }
                // __fmul_rn: the H128 below adds these products; an FMA contracted in one unrolled copy and not in
                // another would round a row by where it sits in the program
                acc[m][jj][r] = __fmul_rn(a, (r & 1) ? d.y : d.x);
            }
        }
        wht128(acc[m], lane);
        const int ra = 32 * wm + 16 * m + gq, rb = ra + 8;   // rows within the program's 64
        const bool oka = rows[ra] >= 0, okb = rows[rb] >= 0;
        const int ia = half_ * gu::BM + ra, ib = ia + 8;      // within the work item's 128
        float sa = 0.f, sb = 0.f;                            // sums of squares of the stored fp16 values
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            half* q = xo + (size_t)(jj >> 1) * PASS * 16 + (jj & 1) * 8;
            const uint32_t va = pack_h2(acc[m][jj][0] * HS, acc[m][jj][1] * HS);
            const uint32_t vb = pack_h2(acc[m][jj][2] * HS, acc[m][jj][3] * HS);
            if (oka) *reinterpret_cast<uint32_t*>(q + ia * 16) = va;
            if (okb) *reinterpret_cast<uint32_t*>(q + ib * 16) = vb;
            const float2 fa = h2f(va), fb = h2f(vb);
            sa = __fmaf_rn(fa.y, fa.y, __fmaf_rn(fa.x, fa.x, sa));
            sb = __fmaf_rn(fb.y, fb.y, __fmaf_rn(fb.x, fb.x, sb));
        }
        if (g.nrm) {                                         // the quad's 4 lanes hold the row's 128 columns
            sa = __fadd_rn(sa, __shfl_xor_sync(0xffffffffu, sa, 1));
            sa = __fadd_rn(sa, __shfl_xor_sync(0xffffffffu, sa, 2));
            sb = __fadd_rn(sb, __shfl_xor_sync(0xffffffffu, sb, 1));
            sb = __fadd_rn(sb, __shfl_xor_sync(0xffffffffu, sb, 2));
            const int G = I >> 7;
            if (t == 0 && oka) g.nrm[(size_t)rows[ra] * G + grp] = sa;
            if (t == 0 && okb) g.nrm[(size_t)rows[rb] * G + grp] = sb;
        }
    }
}

// grid (2 * I / 128, work items): x = column group * 2 + which 64 of the item's members
template <int CB, int NS>
__global__ void __launch_bounds__(gu::NTHR, 2) gu_kernel(GuArgs g) {
    const int2 it = g.items[blockIdx.y];
    if (it.x < 0) return;
    extern __shared__ __align__(16) char sm[];
    const int e = g.uids[it.x];
    const int half_ = blockIdx.x & 1;
    {
        int* rows = reinterpret_cast<int*>(sm);
        for (int i = threadIdx.x; i < gu::BM; i += gu::NTHR) {
            const int m = it.y * PASS + half_ * gu::BM + i;
            const int code = m < g.maxm ? g.members[(size_t)it.x * g.maxm + m] : -1;
            rows[i] = code >= 0 ? (code >> 5) * g.slots + (code & 31) : -1;
            rows[gu::BM + i] = code >= 0 ? (code >> 5) : -1;
        }
    }
    __syncthreads();
    if (reinterpret_cast<const int*>(sm)[0] < 0) return;
    switch (g.k2[e]) {
        case 4: gu_body<CB, 4, NS>(sm, g, e, half_); break;     // 2-bit (GLM-5.3 mosaic)
        case 6: gu_body<CB, 6, NS>(sm, g, e, half_); break;     // 3-bit
        case 8: gu_body<CB, 8, NS>(sm, g, e, half_); break;
        case 12: if constexpr (NS <= gu::depth(12)) gu_body<CB, 12, NS>(sm, g, e, half_); else __trap(); break;
        case 16: if constexpr (NS <= gu::depth(16)) gu_body<CB, 16, NS>(sm, g, e, half_); else __trap(); break;
        default: __trap();
    }
}

// ------------------------------------------------------------------------------------------------ down

namespace dn {
constexpr int WARPS = 8, NTHR = WARPS * 32;
constexpr int NT = 8;                       // tiles a k16 step (128 columns: one a warp)
constexpr int ROWS = 3 * PASS * 4;          // pair index, row and slot weight of the item's members
constexpr int BD = 2 * NT * 512;
constexpr int AST = PASS * 16 * 2;          // an A stage: 128 rows x 16 halves
__host__ __device__ constexpr int wst(int k2) { return NT * 4 * k2; }
__host__ __device__ constexpr size_t smem(int k2, int ns) { return ROWS + BD + (size_t)ns * AST + (size_t)ns * wst(k2) * 4; }
__host__ __device__ constexpr int depth(int k2) { return (int)((SMEM_PAIR - ROWS - BD) / (AST + wst(k2) * 4)); }
}  // namespace dn

struct DnArgs {
    const half* xd;                 // [work items, I / 16, 128, 16] (gu_kernel's)
    const int64_t* tp;              // down trellis pointers [E]
    const int* k2;
    const half* svh;                // [E, D]
    const int* uids;
    const int* ucount;
    const int* members;
    const int* work;
    const int2* items;
    const float* wts;               // [P] (summed)
    const float* grid;              // optional [R] (summed): each row's power-of-two grid; the slots' terms are rounded
                                    // to it, so the fp32 red.adds are exact and their order cannot change out's bits
    float* out;                     // summed: [R, D] (+=); else y [P, D]
    int D, I, maxm, slots;
    int summed;
    int rows = 0;                   // R (summed): with out_budget, the column group a launch sweeps at a time
    long long out_budget = 1ll << 62;
};

__device__ __forceinline__ void dn_epilogue(const DnArgs& g, const int* rows, float (&acc)[16][4], int e, int cb, int wm,
                                            int lane) {
    const int t = lane & 3, gq = lane >> 2;
    const int D = g.D;
    const int col0 = cb * 128 + 2 * t;
    uint32_t sv[16];
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) sv[jj] = __ldg(reinterpret_cast<const unsigned int*>(g.svh + (size_t)e * D + col0 + 8 * jj));
    const int pa = rows[16 * wm + gq], pb = rows[16 * wm + gq + 8];
    float wa = HS, wb = HS;
    float *oa = nullptr, *ob = nullptr;
    float ga = 0.f, gb = 0.f, ia = 1.f, ib = 1.f;     // the rows' grids (0: none) and their inverses (exact: powers of two)
    if (g.summed) {
        const float* wt = reinterpret_cast<const float*>(rows + 2 * PASS);
        if (pa >= 0) { wa = wt[16 * wm + gq] * HS; oa = g.out + (size_t)rows[PASS + 16 * wm + gq] * D + col0; }
        if (pb >= 0) { wb = wt[16 * wm + gq + 8] * HS; ob = g.out + (size_t)rows[PASS + 16 * wm + gq + 8] * D + col0; }
        if (g.grid) {
            if (pa >= 0) { ga = g.grid[rows[PASS + 16 * wm + gq]]; ia = 1.f / ga; }
            if (pb >= 0) { gb = g.grid[rows[PASS + 16 * wm + gq + 8]]; ib = 1.f / gb; }
        }
    } else {
        if (pa >= 0) oa = g.out + (size_t)pa * D + col0;
        if (pb >= 0) ob = g.out + (size_t)pb * D + col0;
    }
    wht128(acc, lane);
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        const float2 s = h2f(sv[jj]);
        const float y0 = acc[jj][0] * s.x, y1 = acc[jj][1] * s.y;
        const float y2 = acc[jj][2] * s.x, y3 = acc[jj][3] * s.y;
        if (g.summed) {
            // the weighted terms, on the rows' grids when given (rintf: an integer multiple of a power of two)
            float t0 = __fmul_rn(y0, wa), t1 = __fmul_rn(y1, wa), t2 = __fmul_rn(y2, wb), t3 = __fmul_rn(y3, wb);
            if (g.grid) {
                t0 = rintf(t0 * ia) * ga; t1 = rintf(t1 * ia) * ga;
                t2 = rintf(t2 * ib) * gb; t3 = rintf(t3 * ib) * gb;
            }
            // lanes t, t ^ 1 swap halves: even t adds row g's columns 2t..2t+3, odd t row g + 8's 2t-2..2t+1
            const bool odd = t & 1;
            const float s0 = odd ? t0 : t2, s1 = odd ? t1 : t3;
            const float r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
            float* o = odd ? ob : oa;
            const float4 v = odd ? make_float4(r0, r1, t2, t3) : make_float4(t0, t1, r0, r1);
            if (o)
                asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(o + 8 * jj - (odd ? 2 : 0)),
                             "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
        } else {
            if (oa) *reinterpret_cast<float2*>(oa + 8 * jj) = make_float2(y0 * wa, y1 * wa);
            if (ob) *reinterpret_cast<float2*>(ob + 8 * jj) = make_float2(y2 * wb, y3 * wb);
        }
    }
}

// Warp wm: rows [16 wm, 16 wm + 16) of the work item, the program's 128 columns.
template <int CB, int K2, int NS>
__device__ __forceinline__ void dn_body(char* sm, const DnArgs& g, int e, int cb) {
    constexpr int TW = 4 * K2, NT = dn::NT, WST = dn::wst(K2);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp;
    const int D = g.D, I = g.I;
    const int* rows = reinterpret_cast<const int*>(sm);
    uint4* bd = reinterpret_cast<uint4*>(sm + dn::ROWS);
    half* as = reinterpret_cast<half*>(sm + dn::ROWS + dn::BD);
    uint32_t* wbuf = reinterpret_cast<uint32_t*>(sm + dn::ROWS + dn::BD + NS * dn::AST);
    const uint32_t* T = reinterpret_cast<const uint32_t*>(g.tp[e]);
    const size_t kstride = (size_t)(D >> 4) * TW;
    const int ct0 = cb * NT;
    const int KS = I / 16;
    const LaneMap<K2> map(lane);
    const bool active = rows[16 * wm] >= 0;

    // this thread's A piece of a k16 slice (4 KB a slice): row i >> 1, 16-byte piece i & 1, swizzled by (row >> 2) & 1
    const int arow = threadIdx.x >> 1, apc = threadIdx.x & 1;
    const bool aok = rows[arow] >= 0;
    const half* asrc = g.xd + (size_t)blockIdx.y * PASS * I + threadIdx.x * 8;
    const int adst = arow * 16 + ((apc ^ ((arow >> 2) & 1)) << 3);
    const WCopy<K2, NT, false, dn::NTHR> wcopy(T, T, ct0);

    float acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;

#pragma unroll
    for (int p = 0; p < NS - 1; ++p) {
        if (p < KS) {
            cp16(as + p * (dn::AST / 2) + adst, asrc + PASS * 16 * p, aok);
            wcopy(wbuf + p * WST, (size_t)p * kstride);
        }
        commit();
    }
    wait_cp<NS - 2>();
    __syncthreads();
    decode_one<CB, K2>(wbuf, bd, warp, lane, map);

    for (int s = 0; s < KS; ++s) {
        wait_cp<NS - 3>();
        __syncthreads();
        if (s + NS - 1 < KS) {
            cp16(as + ((s + NS - 1) % NS) * (dn::AST / 2) + adst, asrc + PASS * 16 * (s + NS - 1), aok);
            wcopy(wbuf + ((s + NS - 1) % NS) * WST, (size_t)(s + NS - 1) * kstride);
        }
        commit();
        if (s + 1 < KS) decode_one<CB, K2>(wbuf + ((s + 1) % NS) * WST, bd + ((s + 1) & 1) * NT * 32, warp, lane, map);
        if (!active) continue;
        const uint4* bs = bd + (s & 1) * NT * 32 + lane;
        const half* a_s = as + (s % NS) * (dn::AST / 2);
        uint32_t a[4];
        {
            const int row = 16 * wm + (lane & 15);
            const int pc = lane >> 4;
            ldsm4(a, a_s + row * 16 + ((pc ^ ((row >> 2) & 1)) << 3));
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint4 b = bs[j * 32];
            mma(acc[2 * j], a, b.x, b.y);
            mma(acc[2 * j + 1], a, b.z, b.w);
        }
    }
    wait_cp<0>();
    if (active) dn_epilogue(g, rows, acc, e, cb, wm, lane);
}

// grid (column blocks a group, work items, column groups)
template <int CB, int NS>
__global__ void __launch_bounds__(dn::NTHR, 2) dn_kernel(DnArgs g) {
    const int2 it = g.items[blockIdx.y];
    if (it.x < 0) return;
    extern __shared__ __align__(16) char sm[];
    const int e = g.uids[it.x];
    {
        int* rows = reinterpret_cast<int*>(sm);
        for (int i = threadIdx.x; i < PASS; i += dn::NTHR) {
            const int m = it.y * PASS + i;
            const int code = m < g.maxm ? g.members[(size_t)it.x * g.maxm + m] : -1;
            const int p = code >= 0 ? (code >> 5) * g.slots + (code & 31) : -1;
            rows[i] = p;
            rows[PASS + i] = code >= 0 ? (code >> 5) : -1;
            if (g.summed) reinterpret_cast<float*>(rows)[2 * PASS + i] = p >= 0 ? __ldg(g.wts + p) : 0.f;
        }
    }
    __syncthreads();
    const int cb = blockIdx.z * gridDim.x + blockIdx.x;
    switch (g.k2[e]) {
        case 4: dn_body<CB, 4, NS>(sm, g, e, cb); break;
        case 6: dn_body<CB, 6, NS>(sm, g, e, cb); break;
        case 8: dn_body<CB, 8, NS>(sm, g, e, cb); break;
        case 12: if constexpr (NS <= dn::depth(12)) dn_body<CB, 12, NS>(sm, g, e, cb); else __trap(); break;
        case 16: if constexpr (NS <= dn::depth(16)) dn_body<CB, 16, NS>(sm, g, e, cb); else __trap(); break;
        default: __trap();
    }
}

static_assert(gu::smem(16, gu::depth(16)) <= SMEM_PAIR && gu::depth(16) >= 4, "two gate/up programs an SM");
static_assert(dn::smem(16, dn::depth(16)) <= SMEM_PAIR && dn::depth(16) >= 4, "two down programs an SM");

template <int CB, int NS>
void gu_go(const GuArgs& g, int items, int k2max, cudaStream_t stream) {
    const size_t smem = gu::smem(k2max, NS);
    C10_CUDA_CHECK(cudaFuncSetAttribute(gu_kernel<CB, NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    dim3 grid((unsigned)(2 * (g.I / 128)), (unsigned)items);
    gu_kernel<CB, NS><<<grid, gu::NTHR, smem, stream>>>(g);
}

// the trellis pipeline as deep as the launch's widest trellis lets two programs an SM go
template <int CB>
void gu_launch(const GuArgs& g, int items, int k2max, cudaStream_t stream) {
    if (k2max <= 8) gu_go<CB, gu::depth(8)>(g, items, k2max, stream);
    else if (k2max <= 12) gu_go<CB, gu::depth(12)>(g, items, k2max, stream);
    else gu_go<CB, gu::depth(16)>(g, items, k2max, stream);
}

template <int CB, int NS>
void dn_go(const DnArgs& g, int items, int k2max, cudaStream_t stream) {
    const size_t smem = dn::smem(k2max, NS);
    C10_CUDA_CHECK(cudaFuncSetAttribute(dn_kernel<CB, NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    // column groups outermost: the programs running at a time add into a slice of out that stays in L2
    const int blocks = g.D / 128;
    int per = blocks;
    while (per > 1 && (size_t)g.rows * per * 128 * 4 > (size_t)g.out_budget && (per % 2) == 0 && blocks % (per / 2) == 0)
        per /= 2;
    dim3 grid((unsigned)per, (unsigned)items, (unsigned)(blocks / per));
    dn_kernel<CB, NS><<<grid, dn::NTHR, smem, stream>>>(g);
}

template <int CB>
void dn_launch(const DnArgs& g, int items, int k2max, cudaStream_t stream) {
    if (k2max <= 12) dn_go<CB, dn::depth(12)>(g, items, k2max, stream);
    else dn_go<CB, dn::depth(16)>(g, items, k2max, stream);
}

}  // namespace tf_exl3p
