#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "mygraph.h"
#include "kfunctions.h"
#include "kfunctions_impl.h"

#include "algs/result.h"
#include "algs/potentialswap.h"
#include "algs/streaming_literature.h"

static long getPeakRSS_KB() {
#if defined(_WIN32)
    return 0;
#else
    struct rusage r;
    if (getrusage(RUSAGE_SELF, &r) == 0) return r.ru_maxrss;
    return 0;
#endif
}

static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static std::string compiled_objective_name() {
#if defined(KFUNC_LT)
    return "lt";
#else
    return "kic";
#endif
}

static std::string mc_option_name(const std::string& objective) {
    return objective == "lt" ? "LT_MC" : "KIC_MC";
}

static std::string seed_option_name(const std::string& objective) {
    return objective == "lt" ? "LT_SEED" : "KIC_SEED";
}

struct ProfitParams {
    std::string model = "suitability";
    double mu = 1.0;
    std::uint64_t suit_seed = 20260528ULL;
    double eta_min = 0.2;
    double eta_max = 0.8;
    double xi_max = 0.2;
};

static bool valid_profit_params(const ProfitParams& p, std::string& error) {
    if (p.mu < 0.0) {
        error = "--profit-mu must be non-negative.";
        return false;
    }
    if (p.eta_min < 0.0 || p.eta_max < 0.0) {
        error = "--eta-min and --eta-max must be non-negative.";
        return false;
    }
    if (p.eta_max < p.eta_min) {
        error = "--eta-max must be at least --eta-min.";
        return false;
    }
    if (p.xi_max < 0.0) {
        error = "--xi-max must be non-negative.";
        return false;
    }
    if (p.mu * p.eta_max > 1.0 + 1e-12) {
        error = "nonnegativity condition failed: profit_mu * eta_max must be <= 1.";
        return false;
    }
    return true;
}

static bool ends_with_ci(const std::string& s, const std::string& suf) {
    if (s.size() < suf.size()) return false;
    const std::size_t off = s.size() - suf.size();
    for (std::size_t i = 0; i < suf.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(s[off + i])) !=
            std::tolower(static_cast<unsigned char>(suf[i]))) {
            return false;
        }
    }
    return true;
}

static bool file_is_empty_or_missing(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return true;
    return st.st_size == 0;
}

static std::string csv_escape(const std::string& s) {
    bool need_quote = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            need_quote = true;
            break;
        }
    }
    if (!need_quote) return s;

    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

static std::vector<std::string> csv_split_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quotes) {
            if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
                cur.push_back('"');
                ++i;
            } else if (c == '"') {
                in_quotes = false;
            } else {
                cur.push_back(c);
            }
        } else if (c == '"') {
            in_quotes = true;
        } else if (c == ',') {
            fields.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    fields.push_back(cur);
    return fields;
}

static bool csv_has_trace_row(const std::string& csv_path,
                              const std::string& graph,
                              const std::string& objective,
                              const ProfitParams& profit_params,
                              const std::string& route,
                              std::size_t B,
                              double alpha,
                              double epsilon,
                              std::size_t t,
                              std::uint64_t seed)
{
    std::ifstream in(csv_path);
    if (!in.good()) return false;

    std::string header_line;
    if (!std::getline(in, header_line)) return false;
    const std::vector<std::string> header = csv_split_line(header_line);

    auto find_idx = [&](const std::string& name) -> int {
        for (std::size_t i = 0; i < header.size(); ++i) {
            if (header[i] == name) return static_cast<int>(i);
        }
        return -1;
    };

    const int graph_i = find_idx("graph");
    const int objective_i = find_idx("objective");
    const int profit_model_i = find_idx("profit_model");
    const int profit_mu_i = find_idx("profit_mu");
    const int suit_seed_i = find_idx("suit_seed");
    const int eta_min_i = find_idx("eta_min");
    const int eta_max_i = find_idx("eta_max");
    const int xi_max_i = find_idx("xi_max");
    const int route_i = find_idx("route");
    const int b_i = find_idx("B");
    const int alpha_i = find_idx("alpha");
    const int epsilon_i = find_idx("epsilon");
    const int t_i = find_idx("t");
    const int seed_i = find_idx("seed");
    if (graph_i < 0 || profit_model_i < 0 || profit_mu_i < 0 ||
        suit_seed_i < 0 || eta_min_i < 0 || eta_max_i < 0 ||
        xi_max_i < 0 || route_i < 0 || b_i < 0 ||
        alpha_i < 0 || epsilon_i < 0 || t_i < 0 || seed_i < 0) {
        return false;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> row = csv_split_line(line);
        const std::size_t need = static_cast<std::size_t>(
            std::max({graph_i, objective_i, profit_model_i, profit_mu_i,
                      suit_seed_i, eta_min_i, eta_max_i, xi_max_i,
                      route_i, b_i, alpha_i, epsilon_i, t_i, seed_i}) + 1);
        if (row.size() < need) continue;
        try {
            const double row_profit_mu = std::stod(row[static_cast<std::size_t>(profit_mu_i)]);
            const double row_eta_min = std::stod(row[static_cast<std::size_t>(eta_min_i)]);
            const double row_eta_max = std::stod(row[static_cast<std::size_t>(eta_max_i)]);
            const double row_xi_max = std::stod(row[static_cast<std::size_t>(xi_max_i)]);
            const double row_alpha = std::stod(row[static_cast<std::size_t>(alpha_i)]);
            const double row_epsilon = std::stod(row[static_cast<std::size_t>(epsilon_i)]);
            const double profit_mu_scale = std::max({1.0, std::fabs(profit_params.mu), std::fabs(row_profit_mu)});
            const double eta_min_scale = std::max({1.0, std::fabs(profit_params.eta_min), std::fabs(row_eta_min)});
            const double eta_max_scale = std::max({1.0, std::fabs(profit_params.eta_max), std::fabs(row_eta_max)});
            const double xi_max_scale = std::max({1.0, std::fabs(profit_params.xi_max), std::fabs(row_xi_max)});
            const double alpha_scale = std::max({1.0, std::fabs(alpha), std::fabs(row_alpha)});
            const double epsilon_scale = std::max({1.0, std::fabs(epsilon), std::fabs(row_epsilon)});
            if (row[static_cast<std::size_t>(graph_i)] == graph &&
                (objective_i < 0 || row[static_cast<std::size_t>(objective_i)] == objective) &&
                row[static_cast<std::size_t>(profit_model_i)] == profit_params.model &&
                std::fabs(row_profit_mu - profit_params.mu) <= 1e-12 * profit_mu_scale &&
                static_cast<std::uint64_t>(std::stoull(row[static_cast<std::size_t>(suit_seed_i)])) == profit_params.suit_seed &&
                std::fabs(row_eta_min - profit_params.eta_min) <= 1e-12 * eta_min_scale &&
                std::fabs(row_eta_max - profit_params.eta_max) <= 1e-12 * eta_max_scale &&
                std::fabs(row_xi_max - profit_params.xi_max) <= 1e-12 * xi_max_scale &&
                row[static_cast<std::size_t>(route_i)] == route &&
                static_cast<std::size_t>(std::stoull(row[static_cast<std::size_t>(b_i)])) == B &&
                std::fabs(row_alpha - alpha) <= 1e-12 * alpha_scale &&
                std::fabs(row_epsilon - epsilon) <= 1e-12 * epsilon_scale &&
                static_cast<std::size_t>(std::stoull(row[static_cast<std::size_t>(t_i)])) == t &&
                static_cast<std::uint64_t>(std::stoull(row[static_cast<std::size_t>(seed_i)])) == seed) {
                return true;
            }
        } catch (...) {
        }
    }
    return false;
}

static void append_csv_row(const std::string& csv_path,
                           const std::string& header,
                           const std::string& row)
{
    if (csv_path.empty()) return;

    const bool write_header = file_is_empty_or_missing(csv_path);
    if (!write_header) {
        std::ifstream existing(csv_path);
        std::string existing_header;
        if (existing.good() && std::getline(existing, existing_header) &&
            existing_header != header)
        {
            std::cerr << "[WARN] CSV header mismatch for " << csv_path
                      << "; refusing to append rows with a different schema.\n";
            return;
        }
    }

    std::ofstream fout(csv_path, std::ios::out | std::ios::app);
    if (!fout.good()) {
        std::cerr << "[WARN] Cannot open CSV for append: " << csv_path << "\n";
        return;
    }
    if (write_header) fout << header << "\n";
    fout << row << "\n";
    fout.flush();
}

static void append_csv_row_unique(const std::string& csv_path,
                                  const std::string& header,
                                  const std::string& row,
                                  const std::string& graph,
                                  const std::string& objective,
                                  const ProfitParams& profit_params,
                                  const std::string& route,
                                  std::size_t B,
                                  double alpha,
                                  double epsilon,
                                  std::size_t t,
                                  std::uint64_t seed)
{
    if (csv_path.empty()) return;
    if (csv_has_trace_row(csv_path, graph, objective, profit_params, route,
                          B, alpha, epsilon, t, seed)) {
        std::cerr << "[INFO] CSV already has " << route
                  << " row at t=" << t << "; skipping duplicate append.\n";
        return;
    }
    append_csv_row(csv_path, header, row);
}

static std::string trace_csv_header() {
    return "graph,objective,profit_model,profit_mu,suit_seed,eta_min,eta_max,xi_max,route,B_factor,B,alpha,epsilon,t,n,m,K,queries,accepted_pairs,swaps,consistency_violations,support_size,f_value,time_sec,graph_mem_mb,algo_mem_mb,seed,kic_mc,decision_f_value,eval_mc";
}

static std::string trace_csv_row(const std::string& graphFile,
                                 const std::string& objective,
                                 const ProfitParams& profit_params,
                                 const std::string& route,
                                 double B_factor,
                                 std::size_t B,
                                 double alpha,
                                 double epsilon,
                                 std::size_t n,
                                 std::size_t m,
                                 std::size_t K,
                                 double graph_mem_mb,
                                 double algo_mem_mb,
                                 std::uint64_t seed,
                                 std::size_t kic_mc,
                                 std::size_t eval_mc,
                                 const algs::PrefixRecord& rec)
{
    std::ostringstream row;
    row << csv_escape(graphFile) << ','
        << csv_escape(objective) << ','
        << csv_escape(profit_params.model) << ','
        << std::setprecision(15) << profit_params.mu << ','
        << profit_params.suit_seed << ','
        << std::setprecision(15) << profit_params.eta_min << ','
        << std::setprecision(15) << profit_params.eta_max << ','
        << std::setprecision(15) << profit_params.xi_max << ','
        << csv_escape(route) << ','
        << std::setprecision(15) << B_factor << ','
        << B << ','
        << std::setprecision(15) << alpha << ','
        << std::setprecision(15) << epsilon << ','
        << rec.t << ','
        << n << ','
        << m << ','
        << K << ','
        << rec.queries << ','
        << rec.accepted_pairs << ','
        << rec.swaps << ','
        << rec.consistency_violations << ','
        << rec.support_size << ','
        << std::setprecision(15) << rec.f_value << ','
        << std::setprecision(15) << rec.time_sec << ','
        << std::setprecision(15) << graph_mem_mb << ','
        << std::setprecision(15) << algo_mem_mb << ','
        << seed << ','
        << kic_mc << ','
        << std::setprecision(15) << rec.decision_f_value << ','
        << eval_mc;
    return row.str();
}

static std::vector<std::size_t> make_report_checkpoints(std::size_t n,
                                                        std::size_t target_count = 20) {
    std::vector<std::size_t> checkpoints;
    if (n == 0) return checkpoints;

    if (n <= target_count) {
        checkpoints.reserve(n);
        for (std::size_t t = 1; t <= n; ++t) checkpoints.push_back(t);
        return checkpoints;
    }

    checkpoints.reserve(target_count);
    const double denom = static_cast<double>(target_count - 1);
    for (std::size_t i = 0; i < target_count; ++i) {
        const double frac = static_cast<double>(i) / denom;
        const std::size_t t = static_cast<std::size_t>(
            std::llround(1.0 + frac * static_cast<double>(n - 1)));
        if (checkpoints.empty() || checkpoints.back() != t) {
            checkpoints.push_back(t);
        }
    }

    if (checkpoints.front() != 1) checkpoints.front() = 1;
    if (checkpoints.back() != n) checkpoints.back() = n;

    std::sort(checkpoints.begin(), checkpoints.end());
    checkpoints.erase(std::unique(checkpoints.begin(), checkpoints.end()), checkpoints.end());
    if (checkpoints.front() != 1) checkpoints.insert(checkpoints.begin(), 1);
    if (checkpoints.back() != n) checkpoints.push_back(n);
    return checkpoints;
}

static void set_env_var(const char* name, const std::string& value) {
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

static void set_profit_env(const ProfitParams& p) {
    set_env_var("KSUB_PROFIT_MU", std::to_string(p.mu));
    set_env_var("KSUB_SUIT_SEED", std::to_string(p.suit_seed));
    set_env_var("KSUB_ETA_MIN", std::to_string(p.eta_min));
    set_env_var("KSUB_ETA_MAX", std::to_string(p.eta_max));
    set_env_var("KSUB_XI_MAX", std::to_string(p.xi_max));
}

static void set_objective_mc_env(const std::string& objective,
                                 std::size_t mc_samples) {
    set_env_var(mc_option_name(objective).c_str(), std::to_string(mc_samples));
}

static void set_objective_seed_env(const std::string& objective,
                                   std::uint64_t seed) {
    set_env_var(seed_option_name(objective).c_str(), std::to_string(seed));
}

static void reevaluate_mc_checkpoints(const mygraph::tinyGraph& g,
                                      algs::Result& res,
                                      const std::string& objective,
                                      std::size_t decision_mc,
                                      std::size_t eval_mc)
{
    if (res.trace.empty()) return;

    for (auto& rec : res.trace) {
        rec.decision_f_value = rec.f_value;
    }

    if (eval_mc == 0) return;

    set_objective_mc_env(objective, eval_mc);
    for (std::size_t i = 0; i < res.trace.size(); ++i) {
        const auto& x = (i < res.x_trace.size() ? res.x_trace[i] : res.x);
        const ksub::PrefixState prefix =
            ksub::make_contiguous_prefix(g.n, res.trace[i].t);
        res.trace[i].f_value = ksub::kfunc_evaluate(g, x, prefix);
    }
    if (!res.trace.empty()) res.f_value = res.trace.back().f_value;
    set_objective_mc_env(objective, decision_mc);
}

static bool is_objective_token(const std::string& tok) {
    const std::string t = to_lower_copy(tok);
    return t == "revenue" || t == "kic" || t == "lt" || t == "sensor";
}

static bool is_route_token(const std::string& tok) {
    const std::string t = to_lower_copy(tok);
    return t == "ops" ||
           t == "potentialswap" ||
           t == "gong" ||
           t == "gong-d1" ||
           t == "pham" ||
           t == "pham22";
}

static std::string normalize_route_token(const std::string& tok) {
    const std::string t = to_lower_copy(tok);
    if (t == "potentialswap") return "ops";
    if (t == "gong-d1") return "gong";
    if (t == "pham22") return "pham";
    return t;
}

static void print_usage(const char* prog) {
    const std::string objective = compiled_objective_name();
    std::cerr
        << "Usage:\n"
        << "  " << prog << " graph.bin route B_factor [options] [csv_path]\n\n"
        << "  objective   : compiled objective is '" << objective
        << "'; optional matching token is accepted\n"
        << "  route       : ops | gong | pham\n"
        << "  B_factor    : budget factor; B = floor(B_factor * |V_n|)\n\n"
        << "Options:\n"
        << "  --alpha <double>     route parameter: OPS d, or baseline alpha\n"
        << "  --epsilon <double>   geometric-guess epsilon for Gong/Pham baselines\n"
        << "  --threads <int>      OpenMP thread count for Monte Carlo objective calls\n"
        << "  --seed <uint64>      seed used for Monte Carlo and recorded in the CSV\n"
        << "  --profit-mu <double> suitability scaling mu (default 1)\n"
        << "  --suit-seed <uint64> seed for deterministic suitability generation\n"
        << "  --eta-min <double>   minimum bad-topic penalty magnitude (default 0.2)\n"
        << "  --eta-max <double>   maximum bad-topic penalty magnitude (default 0.8)\n"
        << "  --xi-max <double>    nonnegative good-topic bonus noise cap (default 0.2)\n"
        << "  --mc <size_t>        decision-oracle Monte Carlo samples (default 200)\n"
        << "  --kic-mc <size_t>    alias for --mc, kept for KIC command compatibility\n"
        << "  --lt-mc <size_t>     alias for --mc, convenient for LT robustness runs\n"
        << "  --eval-mc <size_t>   checkpoint evaluation Monte Carlo samples (default 10000)\n"
        << "  --csv <path>         append prefix-wise trace rows to a CSV file\n"
        << "  --state <path>       save/load resumable state for GongD1/Pham22\n"
        << "  csv_path             optional bare CSV path as the last positional argument\n";
}

int main(int argc, char** argv) {
    using namespace std;
    using namespace mygraph;

    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const string graphFile = argv[1];
    std::size_t argi = 2;
    const string objective = compiled_objective_name();
    if (argi < static_cast<std::size_t>(argc) && is_objective_token(argv[argi])) {
        const string obj_arg = to_lower_copy(argv[argi]);
        if (obj_arg != objective) {
            cerr << "Error: this binary was built for objective '" << objective
                 << "', but command line requested '" << obj_arg << "'.\n";
            return 1;
        }
        ++argi;
    }

    if (argi + 1 >= static_cast<std::size_t>(argc)) {
        print_usage(argv[0]);
        return 1;
    }

    const string route_arg = to_lower_copy(argv[argi]);
    if (!is_route_token(route_arg)) {
        cerr << "Error: route must be ops, gong, or pham.\n";
        return 1;
    }
    const string route = normalize_route_token(route_arg);

    double B_factor = 0.0;
    try {
        B_factor = std::stod(argv[argi + 1]);
    } catch (...) {
        cerr << "Error: invalid budget factor B_factor.\n";
        return 1;
    }
    if (B_factor < 0.0) {
        cerr << "Error: B_factor must be non-negative.\n";
        return 1;
    }

    double alpha = 0.0;
    if (route == "ops") {
        alpha = 1.505;
    } else if (route == "gong" || route == "pham") {
        alpha = 0.4;
    }
    double epsilon = 0.1;
    std::uint64_t seed = 42ULL;
    ProfitParams profit_params;
    std::size_t kic_mc = 200;
    std::size_t eval_mc = 10000;
    int omp_threads = 0;
    string csv_path;
    string state_path;

    for (std::size_t i = argi + 2; i < static_cast<std::size_t>(argc); ++i) {
        string tok = argv[i];
        if (tok == "--alpha") {
            if (i + 1 >= static_cast<std::size_t>(argc)) {
                cerr << "Error: missing value for --alpha\n";
                return 1;
            }
            alpha = std::strtod(argv[++i], nullptr);
        } else if (tok == "--seed") {
            if (i + 1 >= static_cast<std::size_t>(argc)) {
                cerr << "Error: missing value for --seed\n";
                return 1;
            }
            seed = static_cast<std::uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (tok == "--kic-mc" || tok == "--lt-mc" || tok == "--mc") {
            if (i + 1 >= static_cast<std::size_t>(argc)) {
                cerr << "Error: missing value for " << tok << "\n";
                return 1;
            }
            kic_mc = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (tok == "--eval-mc") {
            if (i + 1 >= static_cast<std::size_t>(argc)) {
                cerr << "Error: missing value for --eval-mc\n";
                return 1;
            }
            eval_mc = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (tok == "--epsilon") {
            if (i + 1 >= static_cast<std::size_t>(argc)) {
                cerr << "Error: missing value for --epsilon\n";
                return 1;
            }
            epsilon = std::strtod(argv[++i], nullptr);
        } else if (tok == "--profit-mu" || tok == "--mu") {
            if (i + 1 >= static_cast<std::size_t>(argc)) {
                cerr << "Error: missing value for " << tok << "\n";
                return 1;
            }
            profit_params.mu = std::strtod(argv[++i], nullptr);
        } else if (tok == "--suit-seed" || tok == "--suitability-seed") {
            if (i + 1 >= static_cast<std::size_t>(argc)) {
                cerr << "Error: missing value for " << tok << "\n";
                return 1;
            }
            profit_params.suit_seed =
                static_cast<std::uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (tok == "--eta-min") {
            if (i + 1 >= static_cast<std::size_t>(argc)) {
                cerr << "Error: missing value for --eta-min\n";
                return 1;
            }
            profit_params.eta_min = std::strtod(argv[++i], nullptr);
        } else if (tok == "--eta-max") {
            if (i + 1 >= static_cast<std::size_t>(argc)) {
                cerr << "Error: missing value for --eta-max\n";
                return 1;
            }
            profit_params.eta_max = std::strtod(argv[++i], nullptr);
        } else if (tok == "--xi-max") {
            if (i + 1 >= static_cast<std::size_t>(argc)) {
                cerr << "Error: missing value for --xi-max\n";
                return 1;
            }
            profit_params.xi_max = std::strtod(argv[++i], nullptr);
        } else if (tok == "--threads" || tok == "--omp-threads") {
            if (i + 1 >= static_cast<std::size_t>(argc)) {
                cerr << "Error: missing value for " << tok << "\n";
                return 1;
            }
            omp_threads = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
        } else if (tok == "--csv") {
            if (i + 1 >= static_cast<std::size_t>(argc)) {
                cerr << "Error: missing value for --csv\n";
                return 1;
            }
            csv_path = argv[++i];
        } else if (tok == "--state" || tok == "--resume-state") {
            if (i + 1 >= static_cast<std::size_t>(argc)) {
                cerr << "Error: missing value for " << tok << "\n";
                return 1;
            }
            state_path = argv[++i];
        } else if (!tok.empty() && tok[0] != '-' && csv_path.empty()) {
            csv_path = tok;
        } else if (ends_with_ci(tok, ".csv") && csv_path.empty()) {
            csv_path = tok;
        } else {
            cerr << "[WARN] Unknown arg ignored: " << tok << "\n";
        }
    }

    std::string profit_error;
    if (!valid_profit_params(profit_params, profit_error)) {
        cerr << "Error: " << profit_error << "\n";
        return 1;
    }
    set_profit_env(profit_params);
    set_objective_seed_env(objective, seed);
    set_objective_mc_env(objective, kic_mc);

    if (omp_threads > 0) {
#if defined(_OPENMP)
        omp_set_num_threads(omp_threads);
#else
        std::cerr << "[WARN] --threads ignored because this binary was not built with OpenMP.\n";
#endif
    }

    tinyGraph g;
    if (!g.read_binary(graphFile)) {
        cerr << "Error: cannot read binary graph from " << graphFile << "\n";
        return 1;
    }
    const long graph_peak_rss_kb = getPeakRSS_KB();

    cout << "Graph loaded: n = " << g.n
         << ", m = " << g.m
         << ", K = " << g.K
         << ", undirected = " << (g.undirected ? "true" : "false")
         << "\n";
    cout << "objective = " << objective << "\n";
    cout << "route = " << route << "\n";
    const std::size_t B = static_cast<std::size_t>(std::floor(B_factor * static_cast<double>(g.n)));
    cout << "B_factor = " << B_factor << "\n";
    cout << "B = floor(B_factor * |V_n|) = " << B << "\n";
    cout << "alpha = " << alpha << "\n";
    cout << "epsilon = " << epsilon << "\n";
    cout << "seed = " << seed << "\n";
    cout << "profit_model = " << profit_params.model << "\n";
    cout << "profit_mu = " << profit_params.mu << "\n";
    cout << "suit_seed = " << profit_params.suit_seed << "\n";
    cout << "eta_min = " << profit_params.eta_min << "\n";
    cout << "eta_max = " << profit_params.eta_max << "\n";
    cout << "xi_max = " << profit_params.xi_max << "\n";
    cout << mc_option_name(objective) << " = " << kic_mc << "\n";
    cout << "EVAL_MC = " << eval_mc << "\n";
    if (!state_path.empty()) cout << "state = " << state_path << "\n";
#if defined(_OPENMP)
    cout << "OpenMP max threads = " << omp_get_max_threads() << "\n";
#endif

    const std::vector<std::size_t> report_checkpoints = make_report_checkpoints(g.n, 20);
    const std::string header = trace_csv_header();
    const bool stream_csv_rows = !csv_path.empty();
    const std::string csv_route_label =
        (route == "ops" ? "OPS" : (route == "gong" ? "GongD1" : "Pham22"));
    algs::CheckpointCallback checkpoint_callback;
    if (stream_csv_rows) {
        checkpoint_callback =
            [&](algs::PrefixRecord& rec, const ksub::Assignment& x) {
                rec.decision_f_value = rec.f_value;
                if (eval_mc > 0) {
                    set_objective_mc_env(objective, eval_mc);
                    const ksub::PrefixState prefix =
                        ksub::make_contiguous_prefix(g.n, rec.t);
                    rec.f_value = ksub::kfunc_evaluate(g, x, prefix);
                    set_objective_mc_env(objective, kic_mc);
                }

                const long current_peak_rss_kb = getPeakRSS_KB();
                const long algo_extra_peak_rss_kb =
                    (current_peak_rss_kb > graph_peak_rss_kb
                        ? current_peak_rss_kb - graph_peak_rss_kb
                        : 0);
                const double graph_mem_mb =
                    static_cast<double>(graph_peak_rss_kb) / 1024.0;
                const double algo_mem_mb =
                    static_cast<double>(algo_extra_peak_rss_kb) / 1024.0;

                append_csv_row_unique(
                    csv_path,
                    header,
                    trace_csv_row(
                        graphFile,
                        objective,
                        profit_params,
                        csv_route_label,
                        B_factor,
                        B,
                        alpha,
                        epsilon,
                        g.n,
                        g.m,
                        g.K,
                        graph_mem_mb,
                        algo_mem_mb,
                        seed,
                        kic_mc,
                        eval_mc,
                        rec),
                    graphFile,
                    objective,
                    profit_params,
                    csv_route_label,
                    B,
                    alpha,
                    epsilon,
                    rec.t,
                    seed);
            };
    }
    const algs::CheckpointCallback* checkpoint_callback_ptr =
        stream_csv_rows ? &checkpoint_callback : nullptr;

    algs::Result res;
    if (route == "ops") {
        res = algs::run_one_consistent_potentialswap(
            g, B, alpha, objective, &report_checkpoints, checkpoint_callback_ptr);
    } else if (route == "gong") {
        res = algs::run_gong_d1_streaming(
            g, B, alpha, epsilon, objective, &report_checkpoints,
            checkpoint_callback_ptr, state_path);
    } else if (route == "pham") {
        res = algs::run_pham22_budgeted_streaming(
            g, B, alpha, epsilon, objective, &report_checkpoints,
            checkpoint_callback_ptr, state_path);
    }

    if (!stream_csv_rows) {
        reevaluate_mc_checkpoints(g, res, objective, kic_mc, eval_mc);
    }

    const long final_peak_rss_kb = getPeakRSS_KB();
    const long algo_extra_peak_rss_kb =
        (final_peak_rss_kb > graph_peak_rss_kb ? final_peak_rss_kb - graph_peak_rss_kb : 0);
    res.graph_mem_mb = static_cast<double>(graph_peak_rss_kb) / 1024.0;
    res.algo_mem_mb = static_cast<double>(algo_extra_peak_rss_kb) / 1024.0;
    res.mem_mb = res.algo_mem_mb;

    cout << fixed << setprecision(6);
    cout << "f_value = " << res.f_value << "\n";
    cout << "queries = " << res.queries << "\n";
    cout << "accepted_pairs = " << res.accepted_pairs << "\n";
    cout << "swaps = " << res.swaps << "\n";
    cout << "consistency_violations = " << res.consistency_violations << "\n";
    cout << "time_sec = " << res.time_sec << "\n";
    cout << "graph_mem_mb = " << res.graph_mem_mb << "\n";
    cout << "algo_mem_mb = " << res.algo_mem_mb << "\n";
    cout << "mem_mb = " << res.mem_mb << "\n";
    cout << "report_rows = " << res.trace.size() << "\n";

    return 0;
}
