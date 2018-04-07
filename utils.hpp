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

    std::vector<std::set<int>> find_components()
    {
        const auto num_of_vertices = static_cast<int>(adj_matrix.size());
        std::set<int> vertices_seen;
        std::vector<std::set<int>> components;
        for (int i = 0; i < num_of_vertices; ++i) {
            if (vertices_seen.find(i) != vertices_seen.end()) continue;

            auto comp = bf_walk(i);
            components.push_back(comp);
            vertices_seen.merge(comp);
        }

        return components;
    }

    std::pair<cv::detail::Graph, int> find_max_span_tree(const std::set<int> & vertices);

private:

    std::set<int> bf_walk(int start_idx)
    {
        const auto num_of_vertices = static_cast<int>(adj_matrix.size());
        auto found = std::set<int>();
        auto queue = std::queue<int>();

        found.insert(start_idx);
        queue.push(start_idx);

        int vertex;
        while (not queue.empty()) {
            vertex = queue.front();
            for (int v : this->adj(vertex)) {
                auto [it, havent_found_yet] = found.insert(v);
                if (havent_found_yet)
                    queue.push(v);
            }
            queue.pop();
        }

        return found;
    }

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
