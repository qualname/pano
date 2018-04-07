#include <iosfwd>
#include <iomanip>
#include <set>
#include <queue>
#include <vector>

#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/util.hpp"


namespace utils {


class AdjacencyMatrix {
public:
    AdjacencyMatrix(const std::vector<std::vector<cv::detail::MatchesInfo>> & matrix, double threshold)
    {
        const auto num_of_vertices = static_cast<int>(matrix.size());
        adj_matrix.resize(num_of_vertices);
        for (int i = 0; i < num_of_vertices; ++i) {
            adj_matrix[i].resize(num_of_vertices);
            for (int j = 0; j < num_of_vertices; ++j) {
                const auto conf = matrix[i][j].confidence;
                if (conf> threshold) adj_matrix[i][j] = conf;
                else                 adj_matrix[i][j] = 0.0;
            }
        }
    }

    std::vector<std::set<int>> find_components();

    std::pair<cv::detail::Graph, int> find_max_span_tree(const std::set<int> & vertices);

private:

    std::vector<int> adj(int vertex)
    {
        const auto num_of_vertices = static_cast<int>(adj_matrix.size());
        std::vector<int> adj;
        for (int i = 0; i < num_of_vertices; ++i) {
            if (adj_matrix[vertex][i] > 0.0)
                adj.push_back(i);
        }
        return adj;
    }

    std::vector<std::vector<double>> adj_matrix;
};


} //namespace utils
