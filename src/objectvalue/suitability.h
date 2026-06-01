#ifndef KSUB_OBJECTVALUE_SUITABILITY_H
#define KSUB_OBJECTVALUE_SUITABILITY_H

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "mygraph.h"

namespace ksub {
namespace detail_suitability {

struct SuitabilityParams {
    double mu = 1.0;
    double eta_min = 0.2;
    double eta_max = 0.8;
    double xi_max = 0.2;
    std::uint64_t seed = 20260528ULL;
};

inline double read_env_double(const char* name, double defv) {
    if (const char* s = std::getenv(name)) {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end && *end == '\0') return v;
    }
    return defv;
}

inline std::uint64_t read_env_u64(const char* name, std::uint64_t defv) {
    if (const char* s = std::getenv(name)) {
        char* end = nullptr;
        const unsigned long long v = std::strtoull(s, &end, 10);
        if (end && *end == '\0') return static_cast<std::uint64_t>(v);
    }
    return defv;
}

inline SuitabilityParams read_params() {
    SuitabilityParams p;
    p.mu = read_env_double("KSUB_PROFIT_MU", p.mu);
    p.eta_min = read_env_double("KSUB_ETA_MIN", p.eta_min);
    p.eta_max = read_env_double("KSUB_ETA_MAX", p.eta_max);
    p.xi_max = read_env_double("KSUB_XI_MAX", p.xi_max);
    p.seed = read_env_u64("KSUB_SUIT_SEED", p.seed);
    return p;
}

inline bool same_params(const SuitabilityParams& a, const SuitabilityParams& b) {
    return a.mu == b.mu &&
           a.eta_min == b.eta_min &&
           a.eta_max == b.eta_max &&
           a.xi_max == b.xi_max &&
           a.seed == b.seed;
}

inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline double unit_interval(std::uint64_t x) {
    return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);
}

struct SuitabilityCache {
    const mygraph::tinyGraph* gptr = nullptr;
    std::size_t n = 0;
    std::size_t K = 0;
    SuitabilityParams params;
    std::vector<double> values; // row-major n x K, topics are zero-based.

    void rebuild(const mygraph::tinyGraph& g, const SuitabilityParams& p) {
        gptr = &g;
        n = g.n;
        K = g.K;
        params = p;
        values.assign(n * K, 0.0);
        if (K == 0) return;

        const double eta_lo = std::min(p.eta_min, p.eta_max);
        const double eta_hi = std::max(p.eta_min, p.eta_max);
        const double xi_hi = std::max(0.0, p.xi_max);

        for (std::size_t u = 0; u < n; ++u) {
            const std::uint64_t base =
                splitmix64(p.seed ^ (0x9e3779b97f4a7c15ULL * (u + 1)));
            const std::size_t bad_topic = static_cast<std::size_t>(base % K);
            const double eta =
                eta_lo + (eta_hi - eta_lo) *
                    unit_interval(splitmix64(base ^ 0xbf58476d1ce4e5b9ULL));

            for (std::size_t topic = 0; topic < K; ++topic) {
                double s = -eta;
                if (topic != bad_topic) {
                    const std::uint64_t topic_seed =
                        base ^ (0x94d049bb133111ebULL * (topic + 1));
                    const double xi = xi_hi * unit_interval(splitmix64(topic_seed));
                    s = eta + xi;
                }
                values[u * K + topic] = s;
            }
        }
    }
};

inline const SuitabilityCache& get_cache(const mygraph::tinyGraph& g) {
    static SuitabilityCache cache;
    const SuitabilityParams p = read_params();
    if (cache.gptr != &g ||
        cache.n != g.n ||
        cache.K != g.K ||
        !same_params(cache.params, p))
    {
        cache.rebuild(g, p);
    }
    return cache;
}

inline double profit_mu() {
    return read_params().mu;
}

inline double value(const mygraph::tinyGraph& g,
                    mygraph::node_id u,
                    std::size_t topic_zero_based)
{
    const auto& cache = get_cache(g);
    if (u >= cache.n || topic_zero_based >= cache.K) return 0.0;
    return cache.values[static_cast<std::size_t>(u) * cache.K + topic_zero_based];
}

} // namespace detail_suitability
} // namespace ksub

#endif // KSUB_OBJECTVALUE_SUITABILITY_H
