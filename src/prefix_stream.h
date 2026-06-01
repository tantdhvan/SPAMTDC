#ifndef KSUB_PREFIX_STREAM_H
#define KSUB_PREFIX_STREAM_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mygraph.h"

namespace ksub {

struct PrefixState {
    std::vector<std::uint8_t> active;
    std::size_t visible = 0;

    void reset(std::size_t n) {
        active.assign(n, 0);
        visible = 0;
    }

    void reveal(mygraph::node_id u) {
        const std::size_t idx = static_cast<std::size_t>(u);
        if (idx >= active.size()) return;
        if (active[idx] == 0) {
            active[idx] = 1;
            ++visible;
        }
    }

    bool contains(mygraph::node_id u) const {
        const std::size_t idx = static_cast<std::size_t>(u);
        return idx < active.size() && active[idx] != 0;
    }

    bool empty() const {
        return visible == 0;
    }

    std::size_t size() const {
        return visible;
    }
};

inline PrefixState make_full_prefix(std::size_t n) {
    PrefixState prefix;
    prefix.active.assign(n, 1);
    prefix.visible = n;
    return prefix;
}

inline PrefixState make_contiguous_prefix(std::size_t n, std::size_t t) {
    PrefixState prefix;
    prefix.active.assign(n, 0);
    const std::size_t m = (t < n ? t : n);
    for (std::size_t i = 0; i < m; ++i) {
        prefix.active[i] = 1;
    }
    prefix.visible = m;
    return prefix;
}

inline bool prefix_contains(const PrefixState& prefix, mygraph::node_id u) {
    return prefix.contains(u);
}

} // namespace ksub

#endif // KSUB_PREFIX_STREAM_H
