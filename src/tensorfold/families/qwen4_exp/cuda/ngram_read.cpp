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

#include <condition_variable>
#include <functional>
#include <mutex>

// Reader threads kept for the process (a batch used to start and join up to ``threads`` new threads, ~20-50 us each,
// on every gather: per stream and round). run(work, k): k helpers and the caller run work; returns when all are done.
// One batch at a time (the caller holds the lock while it runs).
class Pool {
public:
    static Pool& get() {
        static Pool* p = new Pool();                 // never destroyed: threads outlive interpreter teardown safely
        return *p;
    }
    void run(const std::function<void()>& work, int k) {
        std::lock_guard<std::mutex> batch(batch_);
        {
            std::unique_lock<std::mutex> l(m_);
            while ((int)threads_.size() < k) threads_.emplace_back([this] { loop(); });
            work_ = &work;
            want_ = k;
            taken_ = 0;
            done_ = 0;
            ++gen_;
        }
        cv_.notify_all();
        work();
        std::unique_lock<std::mutex> l(m_);
        done_cv_.wait(l, [&] { return done_ == want_; });
        work_ = nullptr;
    }

private:
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void()>* w;
            {
                std::unique_lock<std::mutex> l(m_);
                cv_.wait(l, [&] { return gen_ != seen && taken_ < want_; });
                seen = gen_;
                ++taken_;
                w = work_;
            }
            (*w)();
            {
                std::lock_guard<std::mutex> l(m_);
                ++done_;
            }
            done_cv_.notify_one();
        }
    }
    std::mutex batch_, m_;
    std::condition_variable cv_, done_cv_;
    std::vector<std::thread> threads_;
    const std::function<void()>* work_ = nullptr;
    int want_ = 0, taken_ = 0, done_ = 0;
    uint64_t gen_ = 0;
};

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
        Pool::get().run(work, (int)t - 1);           // t - 1 pooled helpers, and this thread
    }
    TORCH_CHECK(bad.load() < 0, "short read of the n-gram table at byte ", bad.load(), ": the pack changed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("pread_runs", &pread_runs); }
