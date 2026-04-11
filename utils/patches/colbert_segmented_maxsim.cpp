// Portable ColBERT segmented_maxsim (std::thread). Upstream uses pthread.h, which MSVC
// does not ship; this file is copied over site-packages/colbert on Windows only.

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>
#include <vector>

typedef struct {
    int tid;
    int nthreads;

    int ndocs;
    int ndoc_vectors;
    int nquery_vectors;

    int64_t* lengths;
    float* scores;
    int64_t* offsets;

    float* max_scores;
} max_args_t;

static void max_worker(void* args) {
    max_args_t* max_args = (max_args_t*)args;

    int ndocs_per_thread =
        (int)std::ceil(((float)max_args->ndocs) / max_args->nthreads);
    int start = max_args->tid * ndocs_per_thread;
    int end = std::min((max_args->tid + 1) * ndocs_per_thread, max_args->ndocs);

    auto max_scores_offset =
        max_args->max_scores + (start * max_args->nquery_vectors);
    auto scores_offset =
        max_args->scores + (max_args->offsets[start] * max_args->nquery_vectors);

    for (int i = start; i < end; i++) {
        for (int j = 0; j < max_args->lengths[i]; j++) {
            std::transform(max_scores_offset,
                           max_scores_offset + max_args->nquery_vectors,
                           scores_offset, max_scores_offset,
                           [](float a, float b) { return std::max(a, b); });
            scores_offset += max_args->nquery_vectors;
        }
        max_scores_offset += max_args->nquery_vectors;
    }
}

torch::Tensor segmented_maxsim(const torch::Tensor scores,
                               const torch::Tensor lengths) {
    auto lengths_a = lengths.data_ptr<int64_t>();
    auto scores_a = scores.data_ptr<float>();
    auto ndocs = lengths.size(0);
    auto ndoc_vectors = scores.size(0);
    auto nquery_vectors = scores.size(1);
    auto nthreads = at::get_num_threads();

    torch::Tensor max_scores =
        torch::zeros({ndocs, nquery_vectors}, scores.options());

    std::vector<int64_t> offs(static_cast<size_t>(ndocs) + 1);
    offs[0] = 0;
    std::partial_sum(lengths_a, lengths_a + ndocs, offs.begin() + 1);

    std::vector<max_args_t> argbuf(static_cast<size_t>(nthreads));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(nthreads));

    for (int i = 0; i < nthreads; i++) {
        argbuf[static_cast<size_t>(i)].tid = i;
        argbuf[static_cast<size_t>(i)].nthreads = nthreads;

        argbuf[static_cast<size_t>(i)].ndocs = ndocs;
        argbuf[static_cast<size_t>(i)].ndoc_vectors = ndoc_vectors;
        argbuf[static_cast<size_t>(i)].nquery_vectors = nquery_vectors;

        argbuf[static_cast<size_t>(i)].lengths = lengths_a;
        argbuf[static_cast<size_t>(i)].scores = scores_a;
        argbuf[static_cast<size_t>(i)].offsets = offs.data();

        argbuf[static_cast<size_t>(i)].max_scores = max_scores.data_ptr<float>();

        threads.emplace_back([i, &argbuf]() {
            max_worker((void*)&argbuf[static_cast<size_t>(i)]);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    return max_scores.sum(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("segmented_maxsim_cpp", &segmented_maxsim, "Segmented MaxSim");
}
