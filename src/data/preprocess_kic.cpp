// preprocess_kic.cpp
// Common binary preprocessor for kIC.
// - Node ids are assigned by first appearance in edges.txt.
// - Therefore node_id == arrival order in the resulting .bin.
// - The output uses the same tinyGraph layout as the other applications.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "mygraph.h"

static inline std::uint64_t pack_uv(std::uint32_t u, std::uint32_t v) {
    return (static_cast<std::uint64_t>(u) << 32) | static_cast<std::uint64_t>(v);
}

static inline std::uint32_t unpack_u(std::uint64_t key) {
    return static_cast<std::uint32_t>(key >> 32);
}

static inline std::uint32_t unpack_v(std::uint64_t key) {
    return static_cast<std::uint32_t>(key & 0xFFFFFFFFULL);
}

static inline double clamp_nonneg(double x) {
    return (x >= 0.0 ? x : 0.0);
}

static void print_usage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog
        << " edges.txt output.bin K [input_undirected(0/1)=1] [randomize_node(0/1)=0] [seed=42]\n"
        << "      [--topic-edge-jitter low high] [--edge-seed seed]\n\n"
        << "edges.txt format:\n"
        << "  - Each line: u v [w] or u v w1 ... wK\n"
        << "  - u,v are original node ids.\n"
        << "  - If one weight is given, it is replicated to all K topics.\n"
        << "    With --topic-edge-jitter, the one weight is multiplied by an\n"
        << "    independent topic-specific factor in [low, high].\n"
        << "  - If K or more weights are given, the first K are used.\n"
        << "  - Negative weights are clamped to 0.\n"
        << "  - Blank lines and lines starting with '#' are ignored.\n\n"
        << "Node ordering:\n"
        << "  - Original ids are renumbered by first appearance in edges.txt.\n"
        << "  - The resulting node_id is the stream arrival order used by the paper code.\n";
}

int main(int argc, char** argv) {
    using namespace std;
    using namespace mygraph;

    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const string edges_txt  = argv[1];
    const string output_bin = argv[2];
    const size_t K = static_cast<size_t>(std::strtoull(argv[3], nullptr, 10));

    if (K == 0) {
        cerr << "Error: K must be > 0.\n";
        return 1;
    }

    bool input_undirected = true;
    bool randomize_node = false;
    unsigned seed = 42;
    bool topic_edge_jitter = false;
    double topic_edge_jitter_low = 1.0;
    double topic_edge_jitter_high = 1.0;
    unsigned edge_seed = 42;

    int argi = 4;
    if (argi < argc && argv[argi][0] != '-') {
        input_undirected = (std::atoi(argv[argi++]) != 0);
    }
    if (argi < argc && argv[argi][0] != '-') {
        randomize_node = (std::atoi(argv[argi++]) != 0);
    }
    if (argi < argc && argv[argi][0] != '-') {
        seed = static_cast<unsigned>(std::strtoul(argv[argi++], nullptr, 10));
    }
    edge_seed = seed;
    for (int i = argi; i < argc; ++i) {
        const string tok = argv[i];
        if (tok == "--topic-edge-jitter") {
            if (i + 2 >= argc) {
                cerr << "Error: --topic-edge-jitter requires low and high.\n";
                return 1;
            }
            topic_edge_jitter = true;
            topic_edge_jitter_low = std::strtod(argv[++i], nullptr);
            topic_edge_jitter_high = std::strtod(argv[++i], nullptr);
            if (topic_edge_jitter_low < 0.0 || topic_edge_jitter_high < topic_edge_jitter_low) {
                cerr << "Error: invalid --topic-edge-jitter range.\n";
                return 1;
            }
        } else if (tok == "--edge-seed") {
            if (i + 1 >= argc) {
                cerr << "Error: --edge-seed requires a seed.\n";
                return 1;
            }
            edge_seed = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
        } else {
            cerr << "Error: unknown argument: " << tok << "\n";
            return 1;
        }
    }

    cout << "Preprocess KIC common binary:\n";
    cout << "  edges.txt         = " << edges_txt << "\n";
    cout << "  output.bin        = " << output_bin << "\n";
    cout << "  K                 = " << K << "\n";
    cout << "  input_undirected  = " << (input_undirected ? "true" : "false") << "\n";
    cout << "  randomize_node    = " << (randomize_node ? "true" : "false") << "\n";
    cout << "  seed              = " << seed << "\n";
    cout << "  topic_edge_jitter = " << (topic_edge_jitter ? "true" : "false") << "\n";
    cout << "  jitter_low        = " << topic_edge_jitter_low << "\n";
    cout << "  jitter_high       = " << topic_edge_jitter_high << "\n";
    cout << "  edge_seed         = " << edge_seed << "\n";
    cout << "  node_id order     = first appearance in edges.txt\n";

    ifstream fin(edges_txt);
    if (!fin) {
        cerr << "Error: cannot open edges file: " << edges_txt << "\n";
        return 1;
    }

    struct RawEdge {
        uint64_t u0;
        uint64_t v0;
        vector<double> w;
    };

    vector<RawEdge> rawEdges;
    rawEdges.reserve(1 << 20);

    unordered_map<uint64_t, node_id> idmap;
    idmap.reserve(1 << 20);

    node_id next_id = 0;
    std::mt19937 edge_gen(edge_seed);
    std::uniform_real_distribution<double> edge_jitter_dist(
        topic_edge_jitter_low, topic_edge_jitter_high);
    auto get_new_id = [&](uint64_t orig) -> node_id {
        auto it = idmap.find(orig);
        if (it != idmap.end()) return it->second;
        const node_id nid = next_id++;
        idmap.emplace(orig, nid);
        return nid;
    };

    string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;

        stringstream ss(line);
        uint64_t u0 = 0, v0 = 0;
        if (!(ss >> u0 >> v0)) continue;

        vector<double> tmp;
        tmp.reserve(K);
        double x = 0.0;
        while (ss >> x) tmp.push_back(clamp_nonneg(x));

        vector<double> w(K, 1.0);
        if (tmp.empty()) {
            if (topic_edge_jitter) {
                for (size_t t = 0; t < K; ++t) {
                    w[t] = edge_jitter_dist(edge_gen);
                }
            }
        } else if (tmp.size() == 1) {
            if (topic_edge_jitter) {
                for (size_t t = 0; t < K; ++t) {
                    w[t] = tmp[0] * edge_jitter_dist(edge_gen);
                }
            } else {
                std::fill(w.begin(), w.end(), tmp[0]);
            }
        } else {
            for (size_t t = 0; t < K && t < tmp.size(); ++t) {
                w[t] = tmp[t];
            }
        }

        rawEdges.push_back({u0, v0, std::move(w)});
        (void)get_new_id(u0);
        (void)get_new_id(v0);
    }
    fin.close();

    const size_t n = static_cast<size_t>(next_id);
    if (n == 0) {
        cerr << "Error: empty graph (no nodes found in edges.txt).\n";
        return 1;
    }

    tinyGraph g;
    g.init(n, K, /*undirected_flag=*/false);
    g.part_id.resize(n);
    for (size_t i = 0; i < n; ++i) {
        g.part_id[i] = static_cast<std::uint32_t>(i);
    }

    if (randomize_node) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(1e-6, 1.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t t = 0; t < K; ++t) {
                g.nodes[i].weights[t] = dist(gen);
                g.nodes[i].alpha[t] = dist(gen);
            }
        }
    }

    unordered_map<uint64_t, vector<double>> keep_first;
    keep_first.reserve(rawEdges.size() * 2 + 1);

    auto try_insert_keep_first = [&](node_id a, node_id b, const vector<double>& ww) {
        const uint64_t key = pack_uv(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
        if (keep_first.find(key) != keep_first.end()) return;
        keep_first.emplace(key, ww);
    };

    for (const auto& re : rawEdges) {
        const node_id u = idmap[re.u0];
        const node_id v = idmap[re.v0];
        try_insert_keep_first(u, v, re.w);
        if (input_undirected) {
            try_insert_keep_first(v, u, re.w);
        }
    }

    vector<uint64_t> keys;
    keys.reserve(keep_first.size());
    for (const auto& kv : keep_first) keys.push_back(kv.first);

    std::sort(keys.begin(), keys.end(), [](uint64_t a, uint64_t b) {
        const uint32_t au = unpack_u(a), av = unpack_v(a);
        const uint32_t bu = unpack_u(b), bv = unpack_v(b);
        if (au != bu) return au < bu;
        return av < bv;
    });

    g.edges.clear();
    g.edges.reserve(keys.size());
    for (uint64_t key : keys) {
        const uint32_t u = unpack_u(key);
        const uint32_t v = unpack_v(key);
        Edge E;
        E.u = static_cast<node_id>(u);
        E.v = static_cast<node_id>(v);
        E.weights = keep_first[key];
        if (E.weights.size() != K) E.weights.resize(K, 0.0);
        g.edges.push_back(std::move(E));
    }
    g.m = g.edges.size();

    // Normalize incoming weights per topic for directed kIC evaluation.
    vector<double> sum_in(g.n * g.K, 0.0);
    for (edge_id eid = 0; eid < g.m; ++eid) {
        const auto& E = g.edges[eid];
        const node_id v = E.v;
        if (v >= g.n) continue;
        for (size_t t = 0; t < g.K; ++t) {
            sum_in[static_cast<size_t>(v) * g.K + t] += E.weights[t];
        }
    }
    for (edge_id eid = 0; eid < g.m; ++eid) {
        auto& E = g.edges[eid];
        const node_id v = E.v;
        if (v >= g.n) continue;
        for (size_t t = 0; t < g.K; ++t) {
            const double s = sum_in[static_cast<size_t>(v) * g.K + t];
            if (s > 0.0) E.weights[t] /= s;
        }
    }

    if (!g.write_binary(output_bin)) {
        cerr << "Error: write_binary failed: " << output_bin << "\n";
        return 1;
    }

    cout << "Done. Saved binary graph to: " << output_bin << "\n";
    cout << "Summary:\n";
    cout << "  n = " << g.n << ", m = " << g.m << ", K = " << g.K
         << ", stored directed = " << (g.undirected ? "true" : "false") << "\n";
    cout << "  node_id == arrival order, part_id[i] = i in the .bin\n";
    return 0;
}
