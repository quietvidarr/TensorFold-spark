#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void gdn_front_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                    const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                    at::Tensor&);
void gdn_back_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, double, at::Tensor&, at::Tensor&,
                   const at::Tensor*, at::Tensor*);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == t, name, ": expected a contiguous CUDA tensor");
}

void front(const at::Tensor& P, const at::Tensor& conv_ptrs, const at::Tensor& sid, const at::Tensor& win,
           const at::Tensor& cw, const at::Tensor& a_log, const at::Tensor& dt_bias, at::Tensor q, at::Tensor k,
           at::Tensor v, at::Tensor g, at::Tensor beta) {
    check(P, at::kBFloat16, "P");
    check(conv_ptrs, at::kLong, "conv pointers");
    check(sid, at::kInt, "stream ids");
    check(win, at::kInt, "windows");
    check(cw, at::kBFloat16, "conv weight");
    check(a_log, at::kFloat, "A_log");
    check(dt_bias, at::kFloat, "dt_bias");
    check(q, at::kFloat, "q");
    check(k, at::kFloat, "k");
    check(v, at::kBFloat16, "v");
    check(g, at::kFloat, "g");
    check(beta, at::kFloat, "beta");
    TORCH_CHECK(win.dim() == 2 && win.size(1) == 4 && sid.numel() == win.size(0) && P.size(0) >= win.size(0),
                "one 4-tap window and one stream id a row");
    c10::cuda::CUDAGuard guard(P.device());
    gdn_front_cuda(P, conv_ptrs, sid, win, cw, a_log, dt_bias, q, k, v, g, beta);
}

// with suh and xh: the rows rotated for an EXL3 projection (rot_in's bits) into xh [rows, value heads * 128] fp16;
// out and xs are then not written
void back(const at::Tensor& y, const at::Tensor& P, const at::Tensor& norm_w, double eps, at::Tensor out,
          at::Tensor xs, c10::optional<at::Tensor> suh, c10::optional<at::Tensor> xh) {
    check(y, at::kBFloat16, "y");
    check(P, at::kBFloat16, "P");
    check(norm_w, at::kBFloat16, "norm");
    check(out, at::kBFloat16, "out");
    check(xs, at::kFloat, "group sums");
    TORCH_CHECK(y.dim() == 3 && y.size(2) == 128 && P.size(0) >= y.size(0), "y is (rows, value heads, 128)");
    c10::cuda::CUDAGuard guard(y.device());
    TORCH_CHECK(suh.has_value() == xh.has_value(), "suh and xh together");
    if (suh) {
        check(*suh, at::kHalf, "suh");
        check(*xh, at::kHalf, "xh");
        TORCH_CHECK(suh->numel() == y.size(1) * 128 && xh->numel() >= y.size(0) * y.size(1) * 128, "suh / xh sizes");
    }
    gdn_back_cuda(y, P, norm_w, eps, out, xs, suh ? &*suh : nullptr, xh ? &*xh : nullptr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("front", &front);
    m.def("back", &back);
}
