// src/objectvalue/kic.h
#ifndef KSUB_OBJECTVALUE_KIC_H
#define KSUB_OBJECTVALUE_KIC_H

#include "objectvalue_common.h"
#include "suitability.h"

#include <vector>
#include <utility>
#include <random>
#include <cstdlib>
#include <cstdint>
#include <algorithm>

#ifndef _OPENMP
#error "KIC requires OpenMP. Please compile with -fopenmp (and link with OpenMP)."
#endif
#include <omp.h>

namespace ksub {

using mygraph::node_id;
using mygraph::edge_id;

namespace detail_kic {

inline double clamp01(double p) {
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return 1.0;
    return p;
}

inline std::size_t read_env_size_t(const char* name, std::size_t defv) {
    if (const char* s = std::getenv(name)) {
        char* end = nullptr;
        unsigned long long v = std::strtoull(s, &end, 10);
        if (end && *end == '\0' && v > 0ULL) return static_cast<std::size_t>(v);
    }
    return defv;
}

inline std::uint64_t read_env_u64(const char* name, std::uint64_t defv) {
    if (const char* s = std::getenv(name)) {
        char* end = nullptr;
        unsigned long long v = std::strtoull(s, &end, 10);
        if (end && *end == '\0') return static_cast<std::uint64_t>(v);
    }
    return defv;
}

inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct KICCache {
    const mygraph::tinyGraph* gptr = nullptr;
    std::size_t n = 0, m = 0, K = 0;
    bool undirected = true;

    std::vector<std::vector<std::vector<std::pair<node_id,double>>>> adjs;

    void rebuild(const mygraph::tinyGraph& g) {
        gptr = &g;
        n = g.n; m = g.m; K = g.K; undirected = g.undirected;

        adjs.assign(K, std::vector<std::vector<std::pair<node_id,double>>>(n));

        for (edge_id eid = 0; eid < g.m; ++eid) {
            const auto& E = g.edges[eid];
            node_id u = E.u;
            node_id v = E.v;
            if (u >= n || v >= n) continue;

            for (std::size_t t = 0; t < K; ++t) {
                double p = (t < E.weights.size() ? E.weights[t] : 0.0);
                p = clamp01(p);
                if (p <= 0.0) continue;

                adjs[t][u].push_back({v, p});

                if (undirected) {
                    adjs[t][v].push_back({u, p});
                }
            }
        }
    }
};

inline const std::vector<std::vector<std::vector<std::pair<node_id,double>>>>&
get_adjs(const mygraph::tinyGraph& g) {
    static KICCache cache;
    if (cache.gptr != &g ||
        cache.n != g.n || cache.m != g.m || cache.K != g.K ||
        cache.undirected != g.undirected)
    {
        cache.rebuild(g);
    }
    return cache.adjs;
}

}


inline double kfunc_evaluate(const mygraph::tinyGraph &g,
                             const Assignment &x,
                             const PrefixState &prefix)
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

    const auto& adjs = detail_kic::get_adjs(g);

    const std::size_t mc = detail_kic::read_env_size_t("KIC_MC", 1000);
    const std::uint64_t base_seed = detail_kic::read_env_u64("KIC_SEED", 42ULL);
    const double mu = detail_suitability::profit_mu();
    if (mc == 0) return mu * suitability_sum;

    double sum_spread = 0.0;

#pragma omp parallel
    {
        std::vector<std::uint32_t> seen_topic(n, 0);
        std::vector<std::uint32_t> seen_union(n, 0);
        std::uint32_t stamp_topic = 1;
        std::uint32_t stamp_union = 1;

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
            std::mt19937_64 rng(detail_kic::splitmix64(base_seed ^ static_cast<std::uint64_t>(it)));

            bump_stamp(stamp_union, seen_union);
            std::size_t union_cnt = 0;

            for (std::size_t t = 0; t < K; ++t) {
                bump_stamp(stamp_topic, seen_topic);

                std::vector<node_id> frontier;
                frontier.reserve(seeds_by_topic[t].size());

                for (node_id s : seeds_by_topic[t]) {
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
                    std::vector<node_id> nxt;
                    nxt.reserve(frontier.size());

                    for (node_id uu : frontier) {
                        const auto& out = adjs[t][uu];
                        for (const auto& vp : out) {
                            node_id v = vp.first;
                            double p = vp.second;
                            if (v >= n) continue;
                            if (!prefix.contains(v)) continue;
                            if (seen_topic[v] == stamp_topic) continue;

                            if (uni(rng) < p) {
                                seen_topic[v] = stamp_topic;
                                nxt.push_back(v);

                                if (seen_union[v] != stamp_union) {
                                    seen_union[v] = stamp_union;
                                    ++union_cnt;
                                }
                            }
                        }
                    }
                    frontier.swap(nxt);
                }
            }

            sum_spread += static_cast<double>(union_cnt);
        }
    }

    const double expected_union = sum_spread / static_cast<double>(mc);
    return expected_union + mu * suitability_sum;
}

inline double kfunc_marginal(const mygraph::tinyGraph &g,
                             node_id u,
                             Label new_label,
                             const Assignment &x,
                             const PrefixState &prefix,
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

inline double kfunc_marginal(const mygraph::tinyGraph &g,
                             node_id u,
                             Label new_label,
                             const Assignment &x,
                             const PrefixState &prefix)
{
    const double fx = kfunc_evaluate(g, x, prefix);
    return kfunc_marginal(g, u, new_label, x, prefix, fx);
}

inline double kfunc_marginal(const mygraph::tinyGraph &g,
                             node_id u,
                             Label new_label,
                             const Assignment &x)
{
    const PrefixState prefix = make_full_prefix(g.n);
    return kfunc_marginal(g, u, new_label, x, prefix);
}

inline double kfunc_evaluate(const mygraph::tinyGraph &g,
                             const Assignment &x)
{
    return kfunc_evaluate(g, x, make_full_prefix(g.n));
}

inline double kfunc_marginal(const mygraph::tinyGraph &g,
                             node_id u,
                             Label new_label,
                             const Assignment &x,
                             double f_x)
{
    return kfunc_marginal(g, u, new_label, x, make_full_prefix(g.n), f_x);
}

}

#endif
