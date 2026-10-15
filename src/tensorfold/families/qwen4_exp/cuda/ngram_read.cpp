// The EXL3 n-gram table's scattered row reads (exl3_pack.NgramTable._pread) in native threads: one pread a run of
// adjacent rows, threads taking runs in turn, the Python lock released for the whole batch. The same bytes as the
// Python loop of os.preadv calls, without paying the interpreter for every one of ~65k runs a 4096-row piece.
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <thread>
#include <vector>

#include <unistd.h>

// fd, file offset, bytes and destination offset (from base) of each run: int64 [n] CPU tensors.
void pread_runs(const at::Tensor& fd, const at::Tensor& offset, const at::Tensor& size, const at::Tensor& dest,
                int64_t base, int64_t threads) {
    const int64_t n = fd.numel();
    TORCH_CHECK(offset.numel() == n && size.numel() == n && dest.numel() == n, "runs: four arrays of one length");
    for (const auto* t : {&fd, &offset, &size, &dest})
        TORCH_CHECK(!t->is_cuda() && t->scalar_type() == at::kLong && t->is_contiguous(), "runs: int64 CPU arrays");
    const int64_t* F = fd.data_ptr<int64_t>();
    const int64_t* O = offset.data_ptr<int64_t>();
    const int64_t* S = size.data_ptr<int64_t>();
    const int64_t* D = dest.data_ptr<int64_t>();
    char* out = reinterpret_cast<char*>(base);
    std::atomic<int64_t> next{0}, bad{-1};
    {
        pybind11::gil_scoped_release nogil;
        auto work = [&] {
            for (;;) {
                const int64_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= n || bad.load(std::memory_order_relaxed) >= 0) return;
                int64_t done = 0;
                while (done < S[i]) {
                    const ssize_t got = ::pread((int)F[i], out + D[i] + done, (size_t)(S[i] - done), (off_t)(O[i] + done));
                    if (got < 0 && errno == EINTR) continue;
                    if (got <= 0) {
                        bad.store(O[i] + done);
                        return;
                    }
                    done += got;
                }
            }
        };
        const int64_t t = std::max<int64_t>(1, std::min<int64_t>(threads, n));
        std::vector<std::thread> pool;
        pool.reserve((size_t)t - 1);
        for (int64_t k = 1; k < t; ++k) pool.emplace_back(work);
        work();
        for (auto& th : pool) th.join();
    }
    TORCH_CHECK(bad.load() < 0, "short read of the n-gram table at byte ", bad.load(), ": the pack changed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("pread_runs", &pread_runs); }
