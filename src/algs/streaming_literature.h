#ifndef KSUB_ALGS_STREAMING_LITERATURE_H
#define KSUB_ALGS_STREAMING_LITERATURE_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "algs/result.h"
#include "algs/stream_utils.h"

namespace algs {

using LiteratureClock = std::chrono::high_resolution_clock;

struct LiteratureGuessState {
    double guess = 0.0;
    ksub::Assignment x;
    ksub::Assignment best_x;
    double value = 0.0;
    double best_value = 0.0;
    std::size_t support = 0;
    std::size_t best_support = 0;
    bool best_matches_current = true;
};

struct LiteratureResumeState {
    std::size_t next_step = 0;
    double elapsed_sec = 0.0;
    Result res;
    std::vector<LiteratureGuessState> states;
    ksub::Assignment best_singleton;
    double best_singleton_seen_value = 0.0;
    bool have_singleton = false;
    ksub::Assignment prev_output;
    bool have_prev_output = false;
};

inline double sanitize_epsilon(double epsilon) {
    if (!(epsilon > 0.0) || !std::isfinite(epsilon)) return 0.1;
    return epsilon;
}

inline bool same_guess(double a, double b) {
    const double scale = std::max({1.0, std::fabs(a), std::fabs(b)});
    return std::fabs(a - b) <= 1e-9 * scale;
}

inline void refresh_geometric_guess_states(std::vector<LiteratureGuessState>& states,
                                           std::size_t n,
                                           double lower_bound,
                                           double upper_bound,
                                           double epsilon)
{
    epsilon = sanitize_epsilon(epsilon);
    if (!(lower_bound > 0.0) || !(upper_bound > 0.0) || upper_bound < lower_bound) {
        states.clear();
        return;
    }

    const double base = 1.0 + epsilon;
    const double log_base = std::log(base);
    const double keep_low = lower_bound / base;
    const double keep_high = upper_bound * base;

    states.erase(std::remove_if(states.begin(), states.end(),
        [&](const LiteratureGuessState& st) {
            return st.guess < keep_low || st.guess > keep_high;
        }), states.end());

    const int j_min = static_cast<int>(std::floor(std::log(lower_bound) / log_base));
    const int j_max = static_cast<int>(std::ceil(std::log(upper_bound) / log_base));

    for (int j = j_min; j <= j_max; ++j) {
        const double guess = std::pow(base, static_cast<double>(j));
        if (guess < keep_low || guess > keep_high) continue;

        bool exists = false;
        for (const auto& st : states) {
            if (same_guess(st.guess, guess)) {
                exists = true;
                break;
            }
        }
        if (exists) continue;

        LiteratureGuessState st;
        st.guess = guess;
        st.x.assign(n, 0);
        st.best_x.assign(n, 0);
        st.best_matches_current = true;
        states.push_back(std::move(st));
    }

    std::sort(states.begin(), states.end(),
        [](const LiteratureGuessState& a, const LiteratureGuessState& b) {
            return a.guess < b.guess;
        });
}

inline void consider_output_candidate(const mygraph::tinyGraph& g,
                                      const ksub::PrefixState& prefix,
                                      const ksub::Assignment& candidate,
                                      std::size_t& query_counter,
                                      double& best_value,
                                      ksub::Assignment& best_x)
{
    const double value = ksub::kfunc_evaluate(g, candidate, prefix);
    ++query_counter;
    if (value > best_value) {
        best_value = value;
        best_x = candidate;
    }
}

inline void consider_known_output_candidate(const ksub::Assignment& candidate,
                                            double value,
                                            double& best_value,
                                            ksub::Assignment& best_x)
{
    if (value > best_value) {
        best_value = value;
        best_x = candidate;
    }
}

inline bool literature_write_assignment(std::ostream& out,
                                        const ksub::Assignment& x)
{
    const std::uint64_t n = static_cast<std::uint64_t>(x.size());
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    if (!x.empty()) {
        out.write(reinterpret_cast<const char*>(x.data()),
                  static_cast<std::streamsize>(x.size() * sizeof(ksub::Label)));
    }
    return out.good();
}

inline bool literature_read_assignment(std::istream& in,
                                       ksub::Assignment& x)
{
    std::uint64_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!in.good()) return false;
    x.assign(static_cast<std::size_t>(n), 0);
    if (!x.empty()) {
        in.read(reinterpret_cast<char*>(x.data()),
                static_cast<std::streamsize>(x.size() * sizeof(ksub::Label)));
    }
    return in.good();
}

inline bool save_literature_state(const std::string& path,
                                  const std::string& route,
                                  std::size_t n,
                                  std::size_t K,
                                  std::size_t B,
                                  double alpha,
                                  double epsilon,
                                  std::size_t next_step,
                                  double elapsed_sec,
                                  const Result& res,
                                  const std::vector<LiteratureGuessState>& states,
                                  const ksub::Assignment& best_singleton,
                                  double best_singleton_seen_value,
                                  bool have_singleton,
                                  const ksub::Assignment& prev_output,
                                  bool have_prev_output)
{
    if (path.empty()) return true;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
        std::cerr << "[WARN] Cannot open state file for write: " << path << "\n";
        return false;
    }

    const char magic[8] = {'K','I','C','R','E','S','1','\0'};
    out.write(magic, sizeof(magic));
    const std::uint64_t version = 1;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    const std::uint64_t route_len = static_cast<std::uint64_t>(route.size());
    out.write(reinterpret_cast<const char*>(&route_len), sizeof(route_len));
    out.write(route.data(), static_cast<std::streamsize>(route.size()));

    const std::uint64_t n64 = static_cast<std::uint64_t>(n);
    const std::uint64_t k64 = static_cast<std::uint64_t>(K);
    const std::uint64_t b64 = static_cast<std::uint64_t>(B);
    const std::uint64_t next64 = static_cast<std::uint64_t>(next_step);
    out.write(reinterpret_cast<const char*>(&n64), sizeof(n64));
    out.write(reinterpret_cast<const char*>(&k64), sizeof(k64));
    out.write(reinterpret_cast<const char*>(&b64), sizeof(b64));
    out.write(reinterpret_cast<const char*>(&alpha), sizeof(alpha));
    out.write(reinterpret_cast<const char*>(&epsilon), sizeof(epsilon));
    out.write(reinterpret_cast<const char*>(&next64), sizeof(next64));
    out.write(reinterpret_cast<const char*>(&elapsed_sec), sizeof(elapsed_sec));

    out.write(reinterpret_cast<const char*>(&res.f_value), sizeof(res.f_value));
    out.write(reinterpret_cast<const char*>(&res.queries), sizeof(res.queries));
    out.write(reinterpret_cast<const char*>(&res.accepted_pairs), sizeof(res.accepted_pairs));
    out.write(reinterpret_cast<const char*>(&res.swaps), sizeof(res.swaps));
    out.write(reinterpret_cast<const char*>(&res.consistency_violations), sizeof(res.consistency_violations));
    literature_write_assignment(out, res.x);

    literature_write_assignment(out, best_singleton);
    out.write(reinterpret_cast<const char*>(&best_singleton_seen_value),
              sizeof(best_singleton_seen_value));
    out.write(reinterpret_cast<const char*>(&have_singleton), sizeof(have_singleton));
    literature_write_assignment(out, prev_output);
    out.write(reinterpret_cast<const char*>(&have_prev_output), sizeof(have_prev_output));

    const std::uint64_t state_count = static_cast<std::uint64_t>(states.size());
    out.write(reinterpret_cast<const char*>(&state_count), sizeof(state_count));
    for (const auto& st : states) {
        out.write(reinterpret_cast<const char*>(&st.guess), sizeof(st.guess));
        out.write(reinterpret_cast<const char*>(&st.value), sizeof(st.value));
        out.write(reinterpret_cast<const char*>(&st.best_value), sizeof(st.best_value));
        out.write(reinterpret_cast<const char*>(&st.support), sizeof(st.support));
        out.write(reinterpret_cast<const char*>(&st.best_support), sizeof(st.best_support));
        out.write(reinterpret_cast<const char*>(&st.best_matches_current),
                  sizeof(st.best_matches_current));
        literature_write_assignment(out, st.x);
        literature_write_assignment(out, st.best_x);
    }

    return out.good();
}

inline bool load_literature_state(const std::string& path,
                                  const std::string& route,
                                  std::size_t n,
                                  std::size_t K,
                                  std::size_t B,
                                  double alpha,
                                  double epsilon,
                                  LiteratureResumeState& loaded)
{
    if (path.empty()) return false;

    std::ifstream in(path, std::ios::binary);
    if (!in.good()) return false;

    char magic[8] = {};
    in.read(magic, sizeof(magic));
    const char expected[8] = {'K','I','C','R','E','S','1','\0'};
    if (!in.good() || !std::equal(magic, magic + 8, expected)) {
        std::cerr << "[WARN] Ignoring incompatible state file: " << path << "\n";
        return false;
    }

    std::uint64_t version = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        std::cerr << "[WARN] Ignoring unsupported state version in: " << path << "\n";
        return false;
    }

    std::uint64_t route_len = 0;
    in.read(reinterpret_cast<char*>(&route_len), sizeof(route_len));
    std::string saved_route(static_cast<std::size_t>(route_len), '\0');
    if (!saved_route.empty()) {
        in.read(&saved_route[0], static_cast<std::streamsize>(saved_route.size()));
    }

    std::uint64_t n64 = 0, k64 = 0, b64 = 0, next64 = 0;
    double saved_alpha = 0.0, saved_epsilon = 0.0;
    in.read(reinterpret_cast<char*>(&n64), sizeof(n64));
    in.read(reinterpret_cast<char*>(&k64), sizeof(k64));
    in.read(reinterpret_cast<char*>(&b64), sizeof(b64));
    in.read(reinterpret_cast<char*>(&saved_alpha), sizeof(saved_alpha));
    in.read(reinterpret_cast<char*>(&saved_epsilon), sizeof(saved_epsilon));
    in.read(reinterpret_cast<char*>(&next64), sizeof(next64));
    in.read(reinterpret_cast<char*>(&loaded.elapsed_sec), sizeof(loaded.elapsed_sec));

    const double alpha_scale = std::max({1.0, std::fabs(alpha), std::fabs(saved_alpha)});
    const double epsilon_scale = std::max({1.0, std::fabs(epsilon), std::fabs(saved_epsilon)});
    const bool compatible =
        saved_route == route &&
        static_cast<std::size_t>(n64) == n &&
        static_cast<std::size_t>(k64) == K &&
        static_cast<std::size_t>(b64) == B &&
        std::fabs(saved_alpha - alpha) <= 1e-12 * alpha_scale &&
        std::fabs(saved_epsilon - epsilon) <= 1e-12 * epsilon_scale &&
        static_cast<std::size_t>(next64) <= n;
    if (!compatible) {
        std::cerr << "[WARN] Ignoring state file with mismatched run parameters: "
                  << path << "\n";
        return false;
    }

    loaded.next_step = static_cast<std::size_t>(next64);
    loaded.res.algo = route;
    loaded.res.objective = "kic";
    loaded.res.route = route;

    in.read(reinterpret_cast<char*>(&loaded.res.f_value), sizeof(loaded.res.f_value));
    in.read(reinterpret_cast<char*>(&loaded.res.queries), sizeof(loaded.res.queries));
    in.read(reinterpret_cast<char*>(&loaded.res.accepted_pairs), sizeof(loaded.res.accepted_pairs));
    in.read(reinterpret_cast<char*>(&loaded.res.swaps), sizeof(loaded.res.swaps));
    in.read(reinterpret_cast<char*>(&loaded.res.consistency_violations),
            sizeof(loaded.res.consistency_violations));
    if (!literature_read_assignment(in, loaded.res.x)) return false;

    if (!literature_read_assignment(in, loaded.best_singleton)) return false;
    in.read(reinterpret_cast<char*>(&loaded.best_singleton_seen_value),
            sizeof(loaded.best_singleton_seen_value));
    in.read(reinterpret_cast<char*>(&loaded.have_singleton), sizeof(loaded.have_singleton));
    if (!literature_read_assignment(in, loaded.prev_output)) return false;
    in.read(reinterpret_cast<char*>(&loaded.have_prev_output), sizeof(loaded.have_prev_output));

    std::uint64_t state_count = 0;
    in.read(reinterpret_cast<char*>(&state_count), sizeof(state_count));
    if (!in.good()) return false;
    loaded.states.clear();
    loaded.states.resize(static_cast<std::size_t>(state_count));
    for (auto& st : loaded.states) {
        in.read(reinterpret_cast<char*>(&st.guess), sizeof(st.guess));
        in.read(reinterpret_cast<char*>(&st.value), sizeof(st.value));
        in.read(reinterpret_cast<char*>(&st.best_value), sizeof(st.best_value));
        in.read(reinterpret_cast<char*>(&st.support), sizeof(st.support));
        in.read(reinterpret_cast<char*>(&st.best_support), sizeof(st.best_support));
        in.read(reinterpret_cast<char*>(&st.best_matches_current),
                sizeof(st.best_matches_current));
        if (!literature_read_assignment(in, st.x)) return false;
        if (!literature_read_assignment(in, st.best_x)) return false;
    }

    return in.good();
}

inline Result run_gong_d1_streaming(const mygraph::tinyGraph& g,
                                    std::size_t B,
                                    double alpha,
                                    double epsilon,
                                    const std::string& objective_name,
                                    const std::vector<std::size_t>* report_checkpoints = nullptr,
                                    const CheckpointCallback* checkpoint_callback = nullptr,
                                    const std::string& state_path = "")
{
    Result res;
    res.algo = "GongD1";
    res.objective = objective_name;
    res.route = "GongD1";

    if (!(alpha > 0.0) || !std::isfinite(alpha)) alpha = 0.4;
    epsilon = sanitize_epsilon(epsilon);

    const std::size_t n = g.n;
    res.x.assign(n, 0);

    ksub::PrefixState prefix;
    prefix.reset(n);

    std::vector<LiteratureGuessState> states;
    ksub::Assignment empty(n, 0);
    ksub::Assignment best_singleton(n, 0);
    double best_singleton_seen_value = 0.0;
    bool have_singleton = false;

    ksub::Assignment prev_output(n, 0);
    bool have_prev_output = false;

    std::size_t start_step = 0;
    double resumed_elapsed_sec = 0.0;
    LiteratureResumeState loaded;
    if (load_literature_state(state_path, "GongD1", n, g.K, B, alpha, epsilon, loaded)) {
        res = loaded.res;
        res.algo = "GongD1";
        res.objective = objective_name;
        res.route = "GongD1";
        states = std::move(loaded.states);
        best_singleton = std::move(loaded.best_singleton);
        best_singleton_seen_value = loaded.best_singleton_seen_value;
        have_singleton = loaded.have_singleton;
        prev_output = std::move(loaded.prev_output);
        have_prev_output = loaded.have_prev_output;
        start_step = loaded.next_step;
        resumed_elapsed_sec = loaded.elapsed_sec;
        std::cerr << "[INFO] Resumed GongD1 from " << state_path
                  << " at prefix " << start_step << ".\n";
    }

    auto t0 = LiteratureClock::now();
    double reporting_overhead_sec = 0.0;

    if (n == 0 || B == 0 || g.K == 0) {
        res.time_sec = 0.0;
        return res;
    }

    for (std::size_t u0 = 0; u0 < start_step && u0 < n; ++u0) {
        prefix.reveal(static_cast<mygraph::node_id>(u0));
    }

    for (std::size_t step = start_step; step < n; ++step) {
        const mygraph::node_id u = static_cast<mygraph::node_id>(step);
        prefix.reveal(u);

        for (auto& st : states) {
            st.value = ksub::kfunc_evaluate(g, st.x, prefix);
            ++res.queries;
            if (st.value > st.best_value) {
                st.best_value = st.value;
                st.best_x = st.x;
                st.best_support = st.support;
                st.best_matches_current = true;
            }
        }

        BestLabelChoice singleton_choice =
            best_label_choice(g, u, empty, prefix, 0.0, res.queries);
        if (singleton_choice.label != 0 &&
            singleton_choice.delta > best_singleton_seen_value) {
            best_singleton.assign(n, 0);
            best_singleton[static_cast<std::size_t>(u)] = singleton_choice.label;
            best_singleton_seen_value = singleton_choice.delta;
            have_singleton = true;
        }

        if (best_singleton_seen_value > 0.0) {
            refresh_geometric_guess_states(
                states,
                n,
                best_singleton_seen_value,
                static_cast<double>(std::max<std::size_t>(1, B)) * best_singleton_seen_value,
                epsilon);
        }

        for (auto& st : states) {
            if (st.support >= B) continue;

            const double threshold =
                alpha * st.guess / static_cast<double>(std::max<std::size_t>(1, B));
            BestLabelChoice choice =
                best_label_choice(g, u, st.x, prefix, st.value, res.queries);
            if (choice.label == 0 || choice.delta < threshold) continue;

            const bool best_was_current = st.best_matches_current;
            st.x[static_cast<std::size_t>(u)] = choice.label;
            st.value += choice.delta;
            ++st.support;
            ++res.accepted_pairs;
            if (best_was_current) st.best_matches_current = false;
            if (st.value > st.best_value) {
                st.best_value = st.value;
                st.best_x = st.x;
                st.best_support = st.support;
                st.best_matches_current = true;
            }
        }

        ksub::Assignment output(n, 0);
        double output_value = 0.0;
        for (auto& st : states) {
            consider_known_output_candidate(st.x, st.value, output_value, output);
            if (!st.best_matches_current && st.best_support > 0) {
                st.best_value = ksub::kfunc_evaluate(g, st.best_x, prefix);
                ++res.queries;
                consider_known_output_candidate(st.best_x, st.best_value,
                                                output_value, output);
            }
        }
        if (have_singleton) {
            consider_output_candidate(g, prefix, best_singleton, res.queries,
                                      output_value, output);
        }

        const std::size_t new_pairs = have_prev_output
            ? new_pair_count(prev_output, output)
            : support_size(output);
        if (new_pairs > 0) ++res.swaps;
        if (new_pairs > 1) res.consistency_violations += (new_pairs - 1);
        prev_output = output;
        have_prev_output = true;

        const double elapsed =
            resumed_elapsed_sec +
            std::chrono::duration<double>(LiteratureClock::now() - t0).count() -
            reporting_overhead_sec;
        res.f_value = output_value;
        if (should_record_prefix(step + 1, report_checkpoints)) {
            PrefixRecord rec{
                step + 1,
                output_value,
                res.queries,
                res.accepted_pairs,
                res.swaps,
                res.consistency_violations,
                std::max(0.0, elapsed),
                support_size(output)
            };
            if (checkpoint_callback) {
                const auto report_start = LiteratureClock::now();
                (*checkpoint_callback)(rec, output);
                reporting_overhead_sec += std::chrono::duration<double>(
                    LiteratureClock::now() - report_start).count();
            }
            res.f_value = rec.f_value;
            res.trace.push_back(rec);
            res.x_trace.push_back(output);
            if (!state_path.empty()) {
                Result state_res = res;
                state_res.x = output;
                save_literature_state(
                    state_path,
                    "GongD1",
                    n,
                    g.K,
                    B,
                    alpha,
                    epsilon,
                    step + 1,
                    rec.time_sec,
                    state_res,
                    states,
                    best_singleton,
                    best_singleton_seen_value,
                    have_singleton,
                    output,
                    have_prev_output);
            }
        }
        res.x.swap(output);
    }

    res.time_sec =
        resumed_elapsed_sec +
        std::chrono::duration<double>(LiteratureClock::now() - t0).count() -
        reporting_overhead_sec;
    if (res.time_sec < 0.0) res.time_sec = 0.0;
    return res;
}

inline Result run_pham22_budgeted_streaming(const mygraph::tinyGraph& g,
                                            std::size_t B,
                                            double alpha,
                                            double epsilon,
                                            const std::string& objective_name,
                                            const std::vector<std::size_t>* report_checkpoints = nullptr,
                                            const CheckpointCallback* checkpoint_callback = nullptr,
                                            const std::string& state_path = "")
{
    Result res;
    res.algo = "Pham22";
    res.objective = objective_name;
    res.route = "Pham22";

    if (!(alpha > 0.0) || !std::isfinite(alpha)) alpha = 0.4;
    epsilon = sanitize_epsilon(epsilon);

    const std::size_t n = g.n;
    res.x.assign(n, 0);

    ksub::PrefixState prefix;
    prefix.reset(n);

    std::vector<LiteratureGuessState> states;
    ksub::Assignment empty(n, 0);
    ksub::Assignment best_singleton(n, 0);
    double best_singleton_seen_value = 0.0;
    bool have_singleton = false;

    ksub::Assignment prev_output(n, 0);
    bool have_prev_output = false;

    std::size_t start_step = 0;
    double resumed_elapsed_sec = 0.0;
    LiteratureResumeState loaded;
    if (load_literature_state(state_path, "Pham22", n, g.K, B, alpha, epsilon, loaded)) {
        res = loaded.res;
        res.algo = "Pham22";
        res.objective = objective_name;
        res.route = "Pham22";
        states = std::move(loaded.states);
        best_singleton = std::move(loaded.best_singleton);
        best_singleton_seen_value = loaded.best_singleton_seen_value;
        have_singleton = loaded.have_singleton;
        prev_output = std::move(loaded.prev_output);
        have_prev_output = loaded.have_prev_output;
        start_step = loaded.next_step;
        resumed_elapsed_sec = loaded.elapsed_sec;
        std::cerr << "[INFO] Resumed Pham22 from " << state_path
                  << " at prefix " << start_step << ".\n";
    }

    auto t0 = LiteratureClock::now();
    double reporting_overhead_sec = 0.0;

    if (n == 0 || B == 0 || g.K == 0) {
        res.time_sec = 0.0;
        return res;
    }

    for (std::size_t u0 = 0; u0 < start_step && u0 < n; ++u0) {
        prefix.reveal(static_cast<mygraph::node_id>(u0));
    }

    for (std::size_t step = start_step; step < n; ++step) {
        const mygraph::node_id u = static_cast<mygraph::node_id>(step);
        prefix.reveal(u);

        for (auto& st : states) {
            st.value = ksub::kfunc_evaluate(g, st.x, prefix);
            ++res.queries;
            if (st.value > st.best_value) {
                st.best_value = st.value;
                st.best_x = st.x;
                st.best_support = st.support;
                st.best_matches_current = true;
            }
        }

        BestLabelChoice singleton_choice =
            best_label_choice(g, u, empty, prefix, 0.0, res.queries);
        if (singleton_choice.label != 0 &&
            singleton_choice.delta > best_singleton_seen_value) {
            best_singleton.assign(n, 0);
            best_singleton[static_cast<std::size_t>(u)] = singleton_choice.label;
            best_singleton_seen_value = singleton_choice.delta;
            have_singleton = true;
        }

        if (best_singleton_seen_value > 0.0) {
            refresh_geometric_guess_states(
                states,
                n,
                best_singleton_seen_value,
                static_cast<double>(std::max<std::size_t>(1, B)) * best_singleton_seen_value,
                epsilon);
        }

        for (auto& st : states) {
            if (st.support >= B) continue;

            BestLabelChoice choice =
                best_label_choice(g, u, st.x, prefix, st.value, res.queries);
            if (choice.label == 0) continue;

            const double after_value = st.value + choice.delta;
            const double after_cost = static_cast<double>(st.support + 1);
            const double density_threshold =
                alpha * st.guess / static_cast<double>(std::max<std::size_t>(1, B));

            if (after_value / after_cost < density_threshold) continue;

            const bool best_was_current = st.best_matches_current;
            st.x[static_cast<std::size_t>(u)] = choice.label;
            st.value = after_value;
            ++st.support;
            ++res.accepted_pairs;
            if (best_was_current) st.best_matches_current = false;
            if (st.value > st.best_value) {
                st.best_value = st.value;
                st.best_x = st.x;
                st.best_support = st.support;
                st.best_matches_current = true;
            }
        }

        ksub::Assignment output(n, 0);
        double output_value = 0.0;
        for (auto& st : states) {
            consider_known_output_candidate(st.x, st.value, output_value, output);
            if (!st.best_matches_current && st.best_support > 0) {
                st.best_value = ksub::kfunc_evaluate(g, st.best_x, prefix);
                ++res.queries;
                consider_known_output_candidate(st.best_x, st.best_value,
                                                output_value, output);
            }
        }
        if (have_singleton) {
            consider_output_candidate(g, prefix, best_singleton, res.queries,
                                      output_value, output);
        }

        const std::size_t new_pairs = have_prev_output
            ? new_pair_count(prev_output, output)
            : support_size(output);
        if (new_pairs > 0) ++res.swaps;
        if (new_pairs > 1) res.consistency_violations += (new_pairs - 1);
        prev_output = output;
        have_prev_output = true;

        const double elapsed =
            resumed_elapsed_sec +
            std::chrono::duration<double>(LiteratureClock::now() - t0).count() -
            reporting_overhead_sec;
        res.f_value = output_value;
        if (should_record_prefix(step + 1, report_checkpoints)) {
            PrefixRecord rec{
                step + 1,
                output_value,
                res.queries,
                res.accepted_pairs,
                res.swaps,
                res.consistency_violations,
                std::max(0.0, elapsed),
                support_size(output)
            };
            if (checkpoint_callback) {
                const auto report_start = LiteratureClock::now();
                (*checkpoint_callback)(rec, output);
                reporting_overhead_sec += std::chrono::duration<double>(
                    LiteratureClock::now() - report_start).count();
            }
            res.f_value = rec.f_value;
            res.trace.push_back(rec);
            res.x_trace.push_back(output);
            if (!state_path.empty()) {
                Result state_res = res;
                state_res.x = output;
                save_literature_state(
                    state_path,
                    "Pham22",
                    n,
                    g.K,
                    B,
                    alpha,
                    epsilon,
                    step + 1,
                    rec.time_sec,
                    state_res,
                    states,
                    best_singleton,
                    best_singleton_seen_value,
                    have_singleton,
                    output,
                    have_prev_output);
            }
        }
        res.x.swap(output);
    }

    res.time_sec =
        resumed_elapsed_sec +
        std::chrono::duration<double>(LiteratureClock::now() - t0).count() -
        reporting_overhead_sec;
    if (res.time_sec < 0.0) res.time_sec = 0.0;
    return res;
}

} // namespace algs

#endif // KSUB_ALGS_STREAMING_LITERATURE_H
