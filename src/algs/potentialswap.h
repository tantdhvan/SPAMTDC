#ifndef KSUB_ALGS_POTENTIALSWAP_H
#define KSUB_ALGS_POTENTIALSWAP_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "algs/result.h"
#include "algs/stream_utils.h"

namespace algs {

using PotentialSwapClock = std::chrono::high_resolution_clock;

struct PotentialSwapPair {
    mygraph::node_id node = 0;
    ksub::Label label = 0;
    double weight = 0.0;
};

inline double potentialswap_threshold(const std::vector<PotentialSwapPair>& selected,
                                      std::size_t B,
                                      double d)
{
    if (B == 0 || selected.empty()) return 0.0;
    if (d <= 0.0) d = 1.505;

    std::vector<double> weights;
    weights.reserve(B);
    for (const auto& item : selected) weights.push_back(item.weight);
    std::sort(weights.begin(), weights.end(), std::greater<double>());
    if (weights.size() < B) weights.resize(B, 0.0);

    const double base = 1.0 + d / static_cast<double>(B);
    const double A = std::pow(base, static_cast<double>(B));
    const double c = (d + 2.0) / (A - 1.0);

    double beta = 0.0;
    double pow_base = 1.0;
    for (std::size_t h = 0; h < B; ++h) {
        const double g_h = (c / static_cast<double>(B)) * pow_base;
        beta += g_h * weights[h];
        pow_base *= base;
    }
    return beta;
}

inline double potentialswap_threshold_from_sorted_weights(const std::vector<double>& weights_desc,
                                                          std::size_t B,
                                                          double d)
{
    if (B == 0 || weights_desc.empty()) return 0.0;
    if (d <= 0.0) d = 1.505;

    const double base = 1.0 + d / static_cast<double>(B);
    const double A = std::pow(base, static_cast<double>(B));
    const double c = (d + 2.0) / (A - 1.0);

    double beta = 0.0;
    double pow_base = 1.0;
    for (std::size_t h = 0; h < B; ++h) {
        const double weight = (h < weights_desc.size() ? weights_desc[h] : 0.0);
        const double g_h = (c / static_cast<double>(B)) * pow_base;
        beta += g_h * weight;
        pow_base *= base;
    }
    return beta;
}

inline void insert_weight_desc(std::vector<double>& weights_desc, double weight) {
    auto it = std::lower_bound(
        weights_desc.begin(),
        weights_desc.end(),
        weight,
        std::greater<double>());
    weights_desc.insert(it, weight);
}

inline void erase_one_weight_desc(std::vector<double>& weights_desc, double weight) {
    for (auto it = weights_desc.begin(); it != weights_desc.end(); ++it) {
        if (std::fabs(*it - weight) <=
            1e-12 * std::max({1.0, std::fabs(*it), std::fabs(weight)})) {
            weights_desc.erase(it);
            return;
        }
    }
}

inline Result run_one_consistent_potentialswap(const mygraph::tinyGraph& g,
                                               std::size_t B,
                                               double d,
                                               const std::string& objective_name,
                                               const std::vector<std::size_t>* report_checkpoints = nullptr,
                                               const CheckpointCallback* checkpoint_callback = nullptr)
{
    Result res;
    res.algo = "OPS";
    res.objective = objective_name;
    res.route = "OPS";

    const std::size_t n = g.n;
    const std::size_t K = g.K;

    res.x.assign(n, 0);
    ksub::Assignment current = res.x;
    std::vector<PotentialSwapPair> selected;
    selected.reserve(B);
    std::vector<double> sorted_weights_desc;
    sorted_weights_desc.reserve(B);
    double beta = 0.0;

    ksub::PrefixState prefix;
    prefix.reset(n);

    auto t0 = PotentialSwapClock::now();
    double reporting_overhead_sec = 0.0;

    if (n == 0) {
        res.time_sec = 0.0;
        return res;
    }

    if (B == 0 || K == 0) {
        for (std::size_t step = 0; step < n; ++step) {
            if (should_record_prefix(step + 1, report_checkpoints)) {
                res.trace.push_back({step + 1, 0.0, 0, 0, 0, 0, 0.0, 0});
                res.x_trace.push_back(current);
            }
        }
        res.time_sec = 0.0;
        return res;
    }

    for (std::size_t step = 0; step < n; ++step) {
        const mygraph::node_id u = static_cast<mygraph::node_id>(step);
        prefix.reveal(u);

        const ksub::Assignment prev_current = current;

        double current_value = ksub::kfunc_evaluate(g, current, prefix);
        ++res.queries;

        BestLabelChoice best;
        best.delta = -std::numeric_limits<double>::infinity();
        best.node = u;
        best = best_label_choice(g, u, current, prefix, current_value, res.queries);

        bool replaced = false;
        if (best.label != 0 && best.delta >= beta) {
            ++res.accepted_pairs;

            if (selected.size() < B) {
                current[static_cast<std::size_t>(u)] = best.label;
                selected.push_back({u, best.label, best.delta});
                insert_weight_desc(sorted_weights_desc, best.delta);
                current_value += best.delta;
            } else {
                std::size_t min_idx = 0;
                double min_weight = selected[0].weight;
                for (std::size_t i = 1; i < selected.size(); ++i) {
                    if (selected[i].weight < min_weight) {
                        min_weight = selected[i].weight;
                        min_idx = i;
                    }
                }

                const mygraph::node_id old_u = selected[min_idx].node;
                current[static_cast<std::size_t>(old_u)] = 0;
                current[static_cast<std::size_t>(u)] = best.label;
                erase_one_weight_desc(sorted_weights_desc, min_weight);
                insert_weight_desc(sorted_weights_desc, best.delta);
                selected[min_idx] = {u, best.label, best.delta};
                replaced = true;
            }
            beta = potentialswap_threshold_from_sorted_weights(sorted_weights_desc, B, d);
        }

        const std::size_t new_pairs = new_pair_count(prev_current, current);
        if (new_pairs > 0) ++res.swaps;
        if (new_pairs > 1) res.consistency_violations += (new_pairs - 1);

        if (replaced) {
            current_value = ksub::kfunc_evaluate(g, current, prefix);
            ++res.queries;
        }
        res.f_value = current_value;

        if (should_record_prefix(step + 1, report_checkpoints)) {
            const double elapsed = std::chrono::duration<double>(
                PotentialSwapClock::now() - t0).count() - reporting_overhead_sec;
            PrefixRecord rec{
                step + 1,
                current_value,
                res.queries,
                res.accepted_pairs,
                res.swaps,
                res.consistency_violations,
                std::max(0.0, elapsed),
                support_size(current)
            };
            if (checkpoint_callback) {
                const auto report_start = PotentialSwapClock::now();
                (*checkpoint_callback)(rec, current);
                reporting_overhead_sec += std::chrono::duration<double>(
                    PotentialSwapClock::now() - report_start).count();
            }
            res.f_value = rec.f_value;
            res.trace.push_back(rec);
            res.x_trace.push_back(current);
        }
    }

    res.x.swap(current);
    res.time_sec = std::chrono::duration<double>(
        PotentialSwapClock::now() - t0).count() - reporting_overhead_sec;
    if (res.time_sec < 0.0) res.time_sec = 0.0;
    return res;
}

} // namespace algs

#endif // KSUB_ALGS_POTENTIALSWAP_H
