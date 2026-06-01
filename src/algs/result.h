// algs/result.h
#ifndef ALGS_RESULT_H
#define ALGS_RESULT_H

#include <string>
#include <vector>
#include <cstddef>
#include <functional>

#include "kfunctions.h"

namespace algs {

struct PrefixRecord {
    std::size_t t = 0;
    double f_value = 0.0;
    std::size_t queries = 0;
    std::size_t accepted_pairs = 0;
    std::size_t swaps = 0;
    std::size_t consistency_violations = 0;
    double time_sec = 0.0;
    std::size_t support_size = 0;
    double decision_f_value = 0.0;
};

struct Result {
    std::string algo;
    std::string objective;
    std::string route;

    double f_value = 0.0;

    std::size_t queries = 0;
    std::size_t accepted_pairs = 0;
    std::size_t swaps = 0;
    std::size_t consistency_violations = 0;

    double time_sec = 0.0;
    double mem_mb = 0.0;
    double graph_mem_mb = 0.0;
    double algo_mem_mb = 0.0;

    ksub::Assignment x;
    std::vector<PrefixRecord> trace;
    std::vector<ksub::Assignment> x_trace;
};

using CheckpointCallback = std::function<void(PrefixRecord&, const ksub::Assignment&)>;

} // namespace algs

#endif // ALGS_RESULT_H
