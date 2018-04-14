#include <vector>

#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/util.hpp"


namespace utils {


struct CmpWeightDesc {
    bool operator()(const std::tuple<int, int, double> & t1,
                    const std::tuple<int, int, double> & t2)
    {
        return std::get<2>(t1) >= std::get<2>(t2);
    }
};


class AdjacencyMatrix {
public:
    AdjacencyMatrix(const std::vector<std::vector<cv::detail::MatchesInfo>> & matrix, double threshold)
    : _max_vertex_idx(static_cast<int>(matrix.size())),
      _span_tree(_max_vertex_idx)
    {
        const auto num_of_vertices = static_cast<int>(matrix.size());
        adj_matrix.resize(num_of_vertices);
        for (int i = 0; i < num_of_vertices; ++i) {
            adj_matrix[i].resize(num_of_vertices);
            for (int j = 0; j < num_of_vertices; ++j) {
                const auto conf = matrix[i][j].confidence;
                if (conf > threshold) adj_matrix[i][j] = conf;
                else                  adj_matrix[i][j] = 0.0;
            }
        }
    }

    void find_max_span_trees()
    {
        auto set = UNION_WHERE(_max_vertex_idx);

        std::set<std::tuple<int, int, double>, CmpWeightDesc> edges;
        for (int v1 = 0; v1 < _max_vertex_idx - 1; ++v1)
            for (int v2 = v1 + 1; v2 < _max_vertex_idx; ++v2)
                edges.insert(std::make_tuple(v1, v2, adj_matrix[v1][v2]));

        int from, to;
        double weight;
        for (const auto & edge : edges) {
            std::tie(from, to, weight) = edge;
            if (weight == 0.0) continue;

            if (set.find_set(from) != set.find_set(to)) {
                _span_tree.addEdge(from, to, static_cast<float>(weight));
                _span_tree.addEdge(to, from, static_cast<float>(weight));
                set.union_(from, to);
            }
        }

        // TODO: find centers
    }

private:
    int _max_vertex_idx;
    cv::detail::Graph _span_tree;
    std::vector<std::set<int>> _components;
    std::vector<std::vector<double>> adj_matrix;
};


} //namespace utils
