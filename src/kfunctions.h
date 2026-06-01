// src/kfunctions.h
#ifndef KFUNCTIONS_H
#define KFUNCTIONS_H

#include <vector>
#include "mygraph.h"
#include "prefix_stream.h"

namespace ksub {

using Label = unsigned char;

using Assignment = std::vector<Label>;

double kfunc_evaluate(const mygraph::tinyGraph &g,
                      const Assignment &x,
                      const PrefixState &prefix);

double kfunc_marginal(const mygraph::tinyGraph &g,
                      mygraph::node_id u,
                      Label new_label,
                      const Assignment &x,
                      const PrefixState &prefix,
                      double f_x);

double kfunc_evaluate(const mygraph::tinyGraph &g,
                      const Assignment &x);

double kfunc_marginal(const mygraph::tinyGraph &g,
                      mygraph::node_id u,
                      Label new_label,
                      const Assignment &x,
                      double f_x);

double kfunc_marginal(const mygraph::tinyGraph &g,
                      mygraph::node_id u,
                      Label new_label,
                      const Assignment &x);

} // namespace ksub

#endif // KFUNCTIONS_H
