#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void p2_to_half_cuda(const at::Tensor&, int64_t, at::Tensor&, int64_t, int64_t, at::Tensor*);
void p2_gate_up_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor*, int64_t,
                     int64_t, int64_t, int64_t, int64_t, int64_t, double, int64_t);
void p2_down_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                  const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor*, const at::Tensor*,
                  at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
void p2_row_grid_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t,
                      int64_t, int64_t, int64_t);
void p2_colmax_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t);
void p2_items_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t);
void p2_slot_sum_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t,
                      int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

// xh [>= R * D] fp16 = x [R, D] (bf16 / fp16, row stride xs) * 2^-2; zero (optional) [R, D] fp32 rows = 0
void to_half(const at::Tensor& x, int64_t xs, at::Tensor xh, int64_t R, int64_t D, c10::optional<at::Tensor> zero) {
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf) && x.stride(1) == 1,
                "x: bf16/fp16 rows");
    TORCH_CHECK(D % 8 == 0 && xs % 8 == 0 && x.data_ptr() != nullptr && (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16) == 0,
                "x: 16-byte aligned rows");
    check(xh, at::kHalf, "xh");
    TORCH_CHECK(xh.numel() >= R * D, "xh too small");
    if (zero)
        TORCH_CHECK(zero->is_cuda() && zero->scalar_type() == at::kFloat && zero->stride(-1) == 1 && zero->stride(0) == D &&
                    zero->size(0) >= R && (reinterpret_cast<uintptr_t>(zero->data_ptr()) % 16) == 0, "zero: fp32 rows of D");
    c10::cuda::CUDAGuard guard(x.device());
    p2_to_half_cuda(x, xs, xh, R, D, zero ? &*zero : nullptr);
}

void gate_up(const at::Tensor& xh, const at::Tensor& tp0, const at::Tensor& tp1, const at::Tensor& k2,
             const at::Tensor& suh0, const at::Tensor& suh1, const at::Tensor& svh0, const at::Tensor& svh1,
             const at::Tensor& suh_d, const at::Tensor& uids, const at::Tensor& ucount, const at::Tensor& members,
             const at::Tensor& work, const at::Tensor& item_tab, at::Tensor xd, int64_t D, int64_t I, int64_t P, int64_t slots, int64_t cb,
             int64_t items, int64_t k2max, double limit, int64_t act_mode, c10::optional<at::Tensor> nrm) {
    check(xh, at::kHalf, "xh");
    check(tp0, at::kLong, "tp0"); check(tp1, at::kLong, "tp1"); check(k2, at::kInt, "k2");
    check(suh0, at::kHalf, "suh0"); check(suh1, at::kHalf, "suh1"); check(svh0, at::kHalf, "svh0");
    check(svh1, at::kHalf, "svh1"); check(suh_d, at::kHalf, "suh_d");
    check(uids, at::kInt, "uids"); check(ucount, at::kInt, "ucount"); check(members, at::kInt, "members");
    check(work, at::kInt, "work"); check(xd, at::kHalf, "xd");
    TORCH_CHECK(D % 128 == 0 && I % 128 == 0, "gate_up: D and I must split in 128");
    TORCH_CHECK(xd.numel() >= items * 128 * I, "xd too small: [work items, I / 16, 128, 16]");
    TORCH_CHECK(k2max == 8 || k2max == 12 || k2max == 16, "k2max 8, 12 or 16");
    if (nrm) { check(*nrm, at::kFloat, "nrm"); TORCH_CHECK(nrm->numel() >= P * (I / 128), "nrm too small: [P, I / 128]"); }
    c10::cuda::CUDAGuard guard(xh.device());
    if (items > 0)
        p2_gate_up_cuda(xh, tp0, tp1, k2, suh0, suh1, svh0, svh1, suh_d, uids, ucount, members, work, item_tab, xd,
                        nrm ? &*nrm : nullptr, D, I, slots, cb, items, k2max, limit, act_mode);
}

// summed (wts given): out [R, D] fp32 += the slots' weighted y (red.add; zero it first); else y [P, D] fp32
void down(const at::Tensor& xd, const at::Tensor& tp, const at::Tensor& k2, const at::Tensor& svh,
          const at::Tensor& uids, const at::Tensor& ucount, const at::Tensor& members, const at::Tensor& work,
          const at::Tensor& item_tab, c10::optional<at::Tensor> wts, at::Tensor out, int64_t D, int64_t I, int64_t slots, int64_t cb,
          int64_t items, int64_t k2max, int64_t rows, int64_t out_budget, c10::optional<at::Tensor> grid) {
    check(xd, at::kHalf, "xd"); check(tp, at::kLong, "tp"); check(k2, at::kInt, "k2"); check(svh, at::kHalf, "svh");
    check(uids, at::kInt, "uids"); check(ucount, at::kInt, "ucount"); check(members, at::kInt, "members");
    check(work, at::kInt, "work");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.stride(-1) == 1 && out.stride(0) == D,
                "out: fp32 rows of D");
    TORCH_CHECK(D % 128 == 0 && I % 16 == 0, "down: D must split in 128, I in 16");
    TORCH_CHECK(k2max == 8 || k2max == 12 || k2max == 16, "k2max 8, 12 or 16");
    if (wts) TORCH_CHECK(wts->is_cuda() && wts->scalar_type() == at::kFloat && wts->is_contiguous(), "wts: fp32");
    if (grid) { TORCH_CHECK(wts, "grid needs wts"); check(*grid, at::kFloat, "grid"); TORCH_CHECK(grid->numel() >= rows, "grid: [R]"); }
    c10::cuda::CUDAGuard guard(xd.device());
    if (items > 0)
        p2_down_cuda(xd, tp, k2, svh, uids, ucount, members, work, item_tab, wts ? &*wts : nullptr, grid ? &*grid : nullptr,
                     out, D, I, slots, cb, items, k2max, rows, wts ? out_budget : 0);
}

// items [n, 2] int32: each work item's (expert slot, pass), -1 past the last (group_any's work list searched once)
void item_table(const at::Tensor& work, const at::Tensor& ucount, at::Tensor items, int64_t n) {
    check(work, at::kInt, "work"); check(ucount, at::kInt, "ucount"); check(items, at::kInt, "items");
    TORCH_CHECK(items.numel() >= 2 * n, "items too small");
    c10::cuda::CUDAGuard guard(work.device());
    if (n > 0) p2_items_cuda(work, ucount, items, n);
}

void slot_sum(const at::Tensor& y, const at::Tensor& pick, const at::Tensor& wts, at::Tensor out, int64_t R, int64_t D,
              int64_t slots, int64_t E) {
    check(y, at::kFloat, "y"); check(pick, at::kInt, "pick"); check(wts, at::kFloat, "wts");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.stride(-1) == 1 && out.stride(0) == D,
                "out: fp32 rows of D");
    TORCH_CHECK(D % 4 == 0, "D % 4");
    c10::cuda::CUDAGuard guard(y.device());
    p2_slot_sum_cuda(y, pick, wts, out, R, D, slots, E);
}

// grid [R] fp32: each row's power-of-two grid for the exact (order-free) summed down (see row_grid_kernel)
void row_grid(const at::Tensor& nrm, const at::Tensor& pick, const at::Tensor& wts, const at::Tensor& colmax, at::Tensor grid,
              int64_t R, int64_t slots, int64_t E, int64_t G) {
    check(nrm, at::kFloat, "nrm"); check(pick, at::kInt, "pick"); check(wts, at::kFloat, "wts");
    check(colmax, at::kFloat, "colmax"); check(grid, at::kFloat, "grid");
    TORCH_CHECK(nrm.numel() >= R * slots * G && grid.numel() >= R && colmax.numel() >= E, "row_grid: sizes");
    c10::cuda::CUDAGuard guard(nrm.device());
    p2_row_grid_cuda(nrm, pick, wts, colmax, grid, R, slots, E, G);
}

// out [n, D / 128] fp32: the largest squared column norm of each 128-column block of W_q [n, I, D] fp16 through
// H128, times svh^2 (svh [n, D] fp16): down_colmax's pieces
void colmax(const at::Tensor& w, const at::Tensor& svh, at::Tensor out, int64_t n, int64_t I, int64_t D) {
    check(w, at::kHalf, "w"); check(svh, at::kHalf, "svh"); check(out, at::kFloat, "out");
    TORCH_CHECK(D % 128 == 0 && w.numel() >= n * I * D && svh.numel() >= n * D && out.numel() >= n * (D / 128),
                "colmax: sizes");
    c10::cuda::CUDAGuard guard(w.device());
    p2_colmax_cuda(w, svh, out, n, I, D);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("to_half", &to_half);
    m.def("gate_up", &gate_up);
    m.def("down", &down);
    m.def("slot_sum", &slot_sum);
    m.def("item_table", &item_table);
    m.def("row_grid", &row_grid);
    m.def("colmax", &colmax);
}
