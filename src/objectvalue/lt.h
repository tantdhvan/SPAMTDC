// src/objectvalue/lt.h
#ifndef KSUB_OBJECTVALUE_LT_H
#define KSUB_OBJECTVALUE_LT_H

#include "objectvalue_common.h"
#include "suitability.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#ifndef _OPENMP
#error "LT requires OpenMP. Please compile with -fopenmp (and link with OpenMP)."
#endif
#include <omp.h>

namespace ksub {

using mygraph::edge_id;
using mygraph::node_id;

namespace detail_lt {

inline double positive_weight(double w) {
    return w > 0.0 ? w : 0.0;
}

inline std::size_t read_env_size_t(const char* primary,
                                   const char* fallback,
                                   std::size_t defv) {
    if (const char* s = std::getenv(primary)) {
        char* end = nullptr;
        unsigned long long v = std::strtoull(s, &end, 10);
        if (end && *end == '\0' && v > 0ULL) return static_cast<std::size_t>(v);
    }
    if (fallback) {
        if (const char* s = std::getenv(fallback)) {
            char* end = nullptr;
            unsigned long long v = std::strtoull(s, &end, 10);
            if (end && *end == '\0' && v > 0ULL) return static_cast<std::size_t>(v);
        }
    }
    return defv;
}

inline std::uint64_t read_env_u64(const char* primary,
                                  const char* fallback,
                                  std::uint64_t defv) {
    if (const char* s = std::getenv(primary)) {
        char* end = nullptr;
        unsigned long long v = std::strtoull(s, &end, 10);
        if (end && *end == '\0') return static_cast<std::uint64_t>(v);
    }
    if (fallback) {
        if (const char* s = std::getenv(fallback)) {
            char* end = nullptr;
            unsigned long long v = std::strtoull(s, &end, 10);
            if (end && *end == '\0') return static_cast<std::uint64_t>(v);
        }
    }
    return defv;
}

inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct LTCache {
    const mygraph::tinyGraph* gptr = nullptr;
    std::size_t n = 0, m = 0, K = 0;
    bool undirected = true;

    std::vector<std::vector<std::vector<std::pair<node_id, double>>>> out_adjs;

    void rebuild(const mygraph::tinyGraph& g) {
        gptr = &g;
        n = g.n;
        m = g.m;
        K = g.K;
        undirected = g.undirected;

        out_adjs.assign(K, std::vector<std::vector<std::pair<node_id, double>>>(n));
        std::vector<std::vector<double>> incoming_sum(K, std::vector<double>(n, 0.0));

        for (edge_id eid = 0; eid < g.m; ++eid) {
            const auto& E = g.edges[eid];
            if (E.u >= n || E.v >= n) continue;
            for (std::size_t t = 0; t < K; ++t) {
                const double w = positive_weight(t < E.weights.size() ? E.weights[t] : 0.0);
                if (w <= 0.0) continue;
                incoming_sum[t][E.v] += w;
                if (undirected) incoming_sum[t][E.u] += w;
            }
        }

        for (edge_id eid = 0; eid < g.m; ++eid) {
            const auto& E = g.edges[eid];
            if (E.u >= n || E.v >= n) continue;
            for (std::size_t t = 0; t < K; ++t) {
                const double raw = positive_weight(t < E.weights.size() ? E.weights[t] : 0.0);
                if (raw <= 0.0) continue;

                const double denom_v = std::max(1.0, incoming_sum[t][E.v]);
                out_adjs[t][E.u].push_back({E.v, raw / denom_v});

                if (undirected) {
                    const double denom_u = std::max(1.0, incoming_sum[t][E.u]);
                    out_adjs[t][E.v].push_back({E.u, raw / denom_u});
                }
            }
        }
    }
};

inline const std::vector<std::vector<std::vector<std::pair<node_id, double>>>>&
get_out_adjs(const mygraph::tinyGraph& g) {
    static LTCache cache;
    if (cache.gptr != &g ||
        cache.n != g.n || cache.m != g.m || cache.K != g.K ||
        cache.undirected != g.undirected)
    {
        cache.rebuild(g);
    }
    return cache.out_adjs;
}

} // namespace detail_lt

inline double kfunc_evaluate(const mygraph::tinyGraph& g,
                             const Assignment& x,
                             const PrefixState& prefix)
{
    const std::size_t n = g.n;
    const std::size_t K = g.K;
    if (n == 0 || K == 0) return 0.0;

    auto label_of = [&](node_id u) -> Label {
        return (u < x.size() ? x[u] : static_cast<Label>(0));
    };

    std::vector<std::vector<node_id>> seeds_by_topic(K);
    double suitability_sum = 0.0;
    for (node_id u = 0; u < static_cast<node_id>(n); ++u) {
        if (!prefix.contains(u)) continue;
        const int lab = static_cast<int>(label_of(u));
        if (1 <= lab && lab <= static_cast<int>(K)) {
            const std::size_t topic = static_cast<std::size_t>(lab - 1);
            seeds_by_topic[topic].push_back(u);
            suitability_sum += detail_suitability::value(g, u, topic);
        }
    }

    const auto& out_adjs = detail_lt::get_out_adjs(g);
    const std::size_t mc = detail_lt::read_env_size_t("LT_MC", "KIC_MC", 1000);
    const std::uint64_t base_seed = detail_lt::read_env_u64("LT_SEED", "KIC_SEED", 42ULL);
    const double mu = detail_suitability::profit_mu();
    if (mc == 0) return mu * suitability_sum;

    double sum_spread = 0.0;

#pragma omp parallel
    {
        std::vector<std::uint32_t> seen_topic(n, 0);
        std::vector<std::uint32_t> seen_union(n, 0);
        std::vector<std::uint32_t> influence_stamp(n, 0);
        std::vector<std::uint32_t> threshold_stamp(n, 0);
        std::vector<double> influence(n, 0.0);
        std::vector<double> threshold(n, 0.0);
        std::uint32_t stamp_topic = 1;
        std::uint32_t stamp_union = 1;
        std::uint32_t stamp_influence = 1;
        std::uint32_t stamp_threshold = 1;

        auto bump_stamp = [](std::uint32_t& st, std::vector<std::uint32_t>& arr) {
            ++st;
            if (st == 0) {
                std::fill(arr.begin(), arr.end(), 0);
                st = 1;
            }
        };

        std::uniform_real_distribution<double> uni(0.0, 1.0);

#pragma omp for schedule(static) reduction(+:sum_spread)
        for (std::size_t it = 0; it < mc; ++it) {
            std::mt19937_64 rng(detail_lt::splitmix64(base_seed ^ static_cast<std::uint64_t>(it)));

            bump_stamp(stamp_union, seen_union);
            std::size_t union_cnt = 0;

            for (std::size_t topic = 0; topic < K; ++topic) {
                bump_stamp(stamp_topic, seen_topic);
                bump_stamp(stamp_influence, influence_stamp);
                bump_stamp(stamp_threshold, threshold_stamp);

                std::vector<node_id> frontier;
                frontier.reserve(seeds_by_topic[topic].size());

                for (node_id s : seeds_by_topic[topic]) {
                    if (s >= n) continue;
                    if (seen_topic[s] == stamp_topic) continue;
                    seen_topic[s] = stamp_topic;
                    frontier.push_back(s);
                    if (seen_union[s] != stamp_union) {
                        seen_union[s] = stamp_union;
                        ++union_cnt;
                    }
                }

                while (!frontier.empty()) {
                    std::vector<node_id> next;
                    next.reserve(frontier.size());

                    for (node_id u : frontier) {
                        for (const auto& vp : out_adjs[topic][u]) {
                            const node_id v = vp.first;
                            const double w = vp.second;
                            if (v >= n) continue;
                            if (!prefix.contains(v)) continue;
                            if (seen_topic[v] == stamp_topic) continue;

                            if (influence_stamp[v] != stamp_influence) {
                                influence_stamp[v] = stamp_influence;
                                influence[v] = 0.0;
                            }
                            influence[v] += w;

                            if (threshold_stamp[v] != stamp_threshold) {
                                threshold_stamp[v] = stamp_threshold;
                                threshold[v] = uni(rng);
                            }

                            if (influence[v] + 1e-12 >= threshold[v]) {
                                seen_topic[v] = stamp_topic;
                                next.push_back(v);
                                if (seen_union[v] != stamp_union) {
                                    seen_union[v] = stamp_union;
                                    ++union_cnt;
                                }
                            }
                        }
                    }

                    frontier.swap(next);
                }
            }

            sum_spread += static_cast<double>(union_cnt);
        }
    }

    const double expected_union = sum_spread / static_cast<double>(mc);
    return expected_union + mu * suitability_sum;
}

inline double kfunc_marginal(const mygraph::tinyGraph& g,
                             node_id u,
                             Label new_label,
                             const Assignment& x,
                             const PrefixState& prefix,
                             double f_x)
{
    const Label old_label = (u < x.size() ? x[u] : static_cast<Label>(0));
    if (!prefix.contains(u)) return 0.0;
    if (old_label == new_label) return 0.0;

    Assignment x_after = x;
    if (u >= x_after.size()) x_after.resize(static_cast<std::size_t>(u) + 1, 0);
    x_after[u] = new_label;

    const double after = kfunc_evaluate(g, x_after, prefix);
    return after - f_x;
}

inline double kfunc_marginal(const mygraph::tinyGraph& g,
                             node_id u,
                             Label new_label,
                             const Assignment& x,
                             const PrefixState& prefix)
{
    const double fx = kfunc_evaluate(g, x, prefix);
    return kfunc_marginal(g, u, new_label, x, prefix, fx);
}

inline double kfunc_marginal(const mygraph::tinyGraph& g,
                             node_id u,
                             Label new_label,
                             const Assignment& x)
{
    const PrefixState prefix = make_full_prefix(g.n);
    return kfunc_marginal(g, u, new_label, x, prefix);
}

inline double kfunc_evaluate(const mygraph::tinyGraph& g,
                             const Assignment& x)
{
    return kfunc_evaluate(g, x, make_full_prefix(g.n));
}

inline double kfunc_marginal(const mygraph::tinyGraph& g,
                             node_id u,
                             Label new_label,
                             const Assignment& x,
                             double f_x)
{
    return kfunc_marginal(g, u, new_label, x, make_full_prefix(g.n), f_x);
}

} // namespace ksub

#endif // KSUB_OBJECTVALUE_LT_H
