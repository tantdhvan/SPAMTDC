#ifndef MYGRAPH_H
#define MYGRAPH_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <sstream>
#include <random>
#include <stdexcept>
#include <algorithm>

namespace mygraph {

using node_id = std::uint32_t;
using edge_id = std::size_t;

struct Edge {
    node_id u;
    node_id v;
    std::vector<double> weights; // size = K
};

struct Node {
    std::vector<double> weights; // size = K
    std::vector<double> alpha;   // size = K
};

struct tinyGraph {
    std::size_t n = 0;
    std::size_t m = 0;
    std::size_t K = 0;
    bool undirected = true;

    std::vector<Node> nodes;
    std::vector<Edge> edges;

    std::vector<std::uint32_t> part_id;

    // indexes
    std::vector<std::vector<edge_id>> incident;
    std::vector<std::vector<edge_id>> incoming;
    std::vector<std::vector<node_id>> neighbors;

    inline void clear();
    inline void init(std::size_t n_nodes,
                     std::size_t k_topics,
                     bool undirected_flag);

    inline void build_index();

    inline edge_id add_edge(node_id u,
                            node_id v,
                            const std::vector<double> &w);

    inline double edge_weight(edge_id e, std::size_t topic) const;

    template <class F>
    inline void for_each_edge(F &&f) const {
        for (edge_id e = 0; e < edges.size(); ++e) {
            const auto &E = edges[e];
            f(E.u, E.v, e);
        }
    }

    template <class F>
    inline void for_each_undirected_edge(F &&f) const {
        if (!undirected) {
            for_each_edge(std::forward<F>(f));
            return;
        }
        for_each_edge(std::forward<F>(f));
    }

    inline bool write_binary(const std::string &path) const;
    inline bool read_binary(const std::string &path);
};

inline bool preprocess_edge_list_to_binary(const std::string &txt_path,
                                          const std::string &bin_path,
                                          std::size_t k_topics,
                                          bool undirected = true,
                                          bool randomize_node = false,
                                          unsigned seed = 42,
                                          const std::string &part_txt_path = "");


inline void tinyGraph::clear() {
    n = m = K = 0;
    undirected = true;
    nodes.clear();
    edges.clear();
    part_id.clear();
    incident.clear();
    incoming.clear();
    neighbors.clear();
}

inline void tinyGraph::init(std::size_t n_nodes,
                            std::size_t k_topics,
                            bool undirected_flag) {
    clear();
    n = n_nodes;
    K = k_topics;
    undirected = undirected_flag;

    nodes.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        nodes[i].weights.assign(K, 1.0);
        nodes[i].alpha.assign(K, 1.0);
    }

    part_id.assign(n, 0);

    incident.assign(n, std::vector<edge_id>());
    incoming.assign(n, std::vector<edge_id>());
    neighbors.assign(n, std::vector<node_id>());
}

inline void tinyGraph::build_index() {
    incident.assign(n, std::vector<edge_id>());
    incoming.assign(n, std::vector<edge_id>());
    neighbors.assign(n, std::vector<node_id>());

    incident.shrink_to_fit();
    incoming.shrink_to_fit();
    neighbors.shrink_to_fit();

    for (edge_id e = 0; e < edges.size(); ++e) {
        const auto &E = edges[e];
        node_id u = E.u, v = E.v;
        if (u >= n || v >= n) continue;

        if (undirected) {
            incident[u].push_back(e);
            incident[v].push_back(e);
            incoming[u].push_back(e);
            incoming[v].push_back(e);
            neighbors[u].push_back(v);
            neighbors[v].push_back(u);
        } else {
            incident[u].push_back(e);   // out-edge
            incoming[v].push_back(e);   // in-edge
            neighbors[u].push_back(v);
        }
    }
}

inline edge_id tinyGraph::add_edge(node_id u,
                                   node_id v,
                                   const std::vector<double> &w) {
    if (w.size() != K) {
        throw std::invalid_argument("add_edge: weight vector size != K");
    }
    if (u >= n || v >= n) {
        throw std::out_of_range("add_edge: node id out of range");
    }

    Edge e;
    e.u = u;
    e.v = v;
    e.weights = w;
    edges.push_back(std::move(e));
    edge_id id = edges.size() - 1;
    ++m;

    if (incident.size() < n) {
        incident.assign(n, std::vector<edge_id>());
        incoming.assign(n, std::vector<edge_id>());
        neighbors.assign(n, std::vector<node_id>());
    }

    if (undirected) {
        incident[u].push_back(id);
        incident[v].push_back(id);
        incoming[u].push_back(id);
        incoming[v].push_back(id);
        neighbors[u].push_back(v);
        neighbors[v].push_back(u);
    } else {
        incident[u].push_back(id);
        incoming[v].push_back(id);
        neighbors[u].push_back(v);
    }

    return id;
}

inline double tinyGraph::edge_weight(edge_id e, std::size_t topic) const {
    if (e >= edges.size()) throw std::out_of_range("edge_weight: edge id out of range");
    if (topic >= K) throw std::out_of_range("edge_weight: topic index out of range");
    return edges[e].weights[topic];
}


inline bool tinyGraph::write_binary(const std::string &path) const {
    if (part_id.size() != n) {
        throw std::invalid_argument("write_binary: part_id.size() != n");
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    std::uint64_t nn = static_cast<std::uint64_t>(n);
    std::uint64_t mm = static_cast<std::uint64_t>(m);
    std::uint64_t KK = static_cast<std::uint64_t>(K);
    std::uint8_t und = undirected ? 1 : 0;

    out.write(reinterpret_cast<const char*>(&nn), sizeof(nn));
    out.write(reinterpret_cast<const char*>(&mm), sizeof(mm));
    out.write(reinterpret_cast<const char*>(&KK), sizeof(KK));
    out.write(reinterpret_cast<const char*>(&und), sizeof(und));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t t = 0; t < K; ++t) {
            double w = nodes[i].weights[t];
            out.write(reinterpret_cast<const char*>(&w), sizeof(double));
        }
        for (std::size_t t = 0; t < K; ++t) {
            double a = nodes[i].alpha[t];
            out.write(reinterpret_cast<const char*>(&a), sizeof(double));
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t pid = static_cast<std::uint32_t>(part_id[i]);
        out.write(reinterpret_cast<const char*>(&pid), sizeof(pid));
    }

    for (const auto &E : edges) {
        out.write(reinterpret_cast<const char*>(&E.u), sizeof(node_id));
        out.write(reinterpret_cast<const char*>(&E.v), sizeof(node_id));
        for (std::size_t t = 0; t < K; ++t) {
            double w = E.weights[t];
            out.write(reinterpret_cast<const char*>(&w), sizeof(double));
        }
    }

    return static_cast<bool>(out);
}

inline bool tinyGraph::read_binary(const std::string &path) {
    clear();
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    std::uint64_t nn = 0, mm = 0, KK = 0;
    std::uint8_t und = 0;

    in.read(reinterpret_cast<char*>(&nn), sizeof(nn));
    in.read(reinterpret_cast<char*>(&mm), sizeof(mm));
    in.read(reinterpret_cast<char*>(&KK), sizeof(KK));
    in.read(reinterpret_cast<char*>(&und), sizeof(und));
    if (!in) return false;

    n = static_cast<std::size_t>(nn);
    m = static_cast<std::size_t>(mm);
    K = static_cast<std::size_t>(KK);
    undirected = (und != 0);

    if (n == 0 || K == 0) return false;

    nodes.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        nodes[i].weights.resize(K);
        nodes[i].alpha.resize(K);

        for (std::size_t t = 0; t < K; ++t) {
            double w = 0.0;
            in.read(reinterpret_cast<char*>(&w), sizeof(double));
            nodes[i].weights[t] = w;
        }
        for (std::size_t t = 0; t < K; ++t) {
            double a = 0.0;
            in.read(reinterpret_cast<char*>(&a), sizeof(double));
            nodes[i].alpha[t] = a;
        }
    }

    part_id.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t pid = 0;
        in.read(reinterpret_cast<char*>(&pid), sizeof(pid));
        part_id[i] = pid;
    }

    edges.resize(m);
    for (std::size_t e = 0; e < m; ++e) {
        node_id u = 0, v = 0;
        in.read(reinterpret_cast<char*>(&u), sizeof(node_id));
        in.read(reinterpret_cast<char*>(&v), sizeof(node_id));
        edges[e].u = u;
        edges[e].v = v;

        edges[e].weights.resize(K);
        for (std::size_t t = 0; t < K; ++t) {
            double w = 0.0;
            in.read(reinterpret_cast<char*>(&w), sizeof(double));
            edges[e].weights[t] = w;
        }
    }

    if (!in) return false;

    build_index();
    return true;
}

inline bool load_part_map_txt_(const std::string& part_txt_path,
                              std::unordered_map<std::uint64_t, std::uint32_t>& partmap)
{
    std::ifstream pin(part_txt_path);
    if (!pin) return false;

    std::string line;
    while (std::getline(pin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::uint64_t u0 = 0, p0 = 0;
        std::stringstream ss(line);
        ss >> u0 >> p0;
        if (!ss) continue;
        partmap[u0] = static_cast<std::uint32_t>(p0);
    }
    return true;
}

inline bool preprocess_edge_list_to_binary(const std::string &txt_path,
                                          const std::string &bin_path,
                                          std::size_t k_topics,
                                          bool undirected,
                                          bool randomize_node,
                                          unsigned seed,
                                          const std::string &part_txt_path)
{
    std::ifstream in(txt_path);
    if (!in) return false;

    std::unordered_map<std::uint64_t, std::uint32_t> partmap;
    if (!part_txt_path.empty()) {
        if (!load_part_map_txt_(part_txt_path, partmap)) return false;
    }

    std::unordered_map<std::uint64_t, node_id> idmap;
    idmap.reserve(1 << 16);

    struct RawEdge {
        std::uint64_t u_orig;
        std::uint64_t v_orig;
        double w;
    };
    std::vector<RawEdge> rawEdges;
    rawEdges.reserve(1 << 20);

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::uint64_t u0 = 0, v0 = 0;
        double w = 1.0;
        std::stringstream ss(line);
        ss >> u0 >> v0;
        if (!ss) continue;
        if (ss >> w) {
            if (w <= 0.0) w = 1e-6;
        }
        rawEdges.push_back({u0, v0, w});
    }
    in.close();

    node_id next_id = 0;
    std::vector<std::uint32_t> part_new;
    part_new.reserve(1 << 16);

    auto get_new_id = [&](std::uint64_t orig) -> node_id {
        auto it = idmap.find(orig);
        if (it != idmap.end()) return it->second;

        node_id nid = next_id++;
        idmap.emplace(orig, nid);

        std::uint32_t pid = 0;
        if (!partmap.empty()) {
            auto itp = partmap.find(orig);
            if (itp != partmap.end()) pid = itp->second;
        }
        part_new.push_back(pid);
        return nid;
    };

    std::vector<Edge> edges;
    edges.reserve(rawEdges.size());

    for (const auto &re : rawEdges) {
        node_id u = get_new_id(re.u_orig);
        node_id v = get_new_id(re.v_orig);

        Edge E;
        E.u = u;
        E.v = v;
        E.weights.assign(k_topics, re.w);
        edges.push_back(std::move(E));
    }

    tinyGraph g;
    g.n = static_cast<std::size_t>(next_id);
    g.m = edges.size();
    g.K = k_topics;
    g.undirected = undirected;

    g.nodes.resize(g.n);
    for (std::size_t i = 0; i < g.n; ++i) {
        g.nodes[i].weights.assign(k_topics, 1.0);
        g.nodes[i].alpha.assign(k_topics, 1.0);
    }

    if (randomize_node) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(1e-6, 1.0);
        for (std::size_t i = 0; i < g.n; ++i) {
            for (std::size_t t = 0; t < k_topics; ++t) {
                g.nodes[i].weights[t] = dist(gen);
                g.nodes[i].alpha[t] = dist(gen);
            }
        }
    }

    g.part_id.assign(g.n, 0);
    if (!part_new.empty()) {
        if (part_new.size() != g.n) return false; // defensive
        for (std::size_t i = 0; i < g.n; ++i) g.part_id[i] = part_new[i];
    }

    g.edges = std::move(edges);

    return g.write_binary(bin_path);
}

}

#endif // MYGRAPH_H
