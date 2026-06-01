#ifndef KSUB_ALGS_STREAM_UTILS_H
#define KSUB_ALGS_STREAM_UTILS_H

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "mygraph.h"
#include "kfunctions.h"
#include "prefix_stream.h"

namespace algs {

struct BestLabelChoice {
    mygraph::node_id node = static_cast<mygraph::node_id>(-1);
    ksub::Label label = 0;
    double delta = -std::numeric_limits<double>::infinity();
};

inline BestLabelChoice best_label_choice(const mygraph::tinyGraph& g,
                                         mygraph::node_id u,
                                         const ksub::Assignment& x,
                                         const ksub::PrefixState& prefix,
                                         double f_x,
                                         std::size_t& query_counter)
{
    BestLabelChoice best;
    best.node = u;

    if (!prefix.contains(u)) return best;

    for (ksub::Label lbl = 1; lbl <= static_cast<ksub::Label>(g.K); ++lbl) {
        const double delta = ksub::kfunc_marginal(g, u, lbl, x, prefix, f_x);
        ++query_counter;
        if (delta > best.delta) {
            best.delta = delta;
            best.label = lbl;
        }
    }

    return best;
}

inline std::size_t support_size(const ksub::Assignment& x) {
    std::size_t cnt = 0;
    for (ksub::Label lbl : x) {
        if (lbl != 0) ++cnt;
    }
    return cnt;
}

inline std::size_t new_pair_count(const ksub::Assignment& prev,
                                  const ksub::Assignment& next)
{
    const std::size_t n = std::min(prev.size(), next.size());
    std::size_t added = 0;
    for (std::size_t u = 0; u < n; ++u) {
        if (next[u] != 0 && prev[u] != next[u]) ++added;
    }
    for (std::size_t u = n; u < next.size(); ++u) {
        if (next[u] != 0) ++added;
    }
    return added;
}

inline bool should_record_prefix(std::size_t t,
                                 const std::vector<std::size_t>* checkpoints)
{
    return checkpoints == nullptr ||
           std::binary_search(checkpoints->begin(), checkpoints->end(), t);
}

} // namespace algs

#endif // KSUB_ALGS_STREAM_UTILS_H
