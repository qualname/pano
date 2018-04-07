#include <iosfwd>
#include <iomanip>
#include <set>
#include <queue>
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


struct Increment {
    Increment(std::map<int, int> & distances_) : distances(distances_) {}
    void operator()(const cv::detail::GraphEdge & edge) {
        distances[edge.to] = distances[edge.from] + 1;
    }

    std::map<int, int> & distances;
};

class AdjacencyMatrix {
public:
    AdjacencyMatrix(const std::vector<std::vector<cv::detail::MatchesInfo>> & matrix, double threshold)
    : components(static_cast<int>(matrix.size()))
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

    std::pair<cv::detail::Graph, int> find_max_span_tree(const std::set<int>     & vertices,
                                                               cv::detail::Graph & span_tree)
    {
        const auto num_of_vertices = static_cast<int>(vertices.size());

        std::set<std::tuple<int, int, double>, CmpWeightDesc> edges;
        for (const auto v1 : vertices)
            for (const auto v2 : vertices)
                if (v1 < v2)
                    edges.insert(std::make_tuple(v1, v2, adj_matrix[v1][v2]));

        int from, to;
        double weight;
        std::map<int, int> power;
        for (const auto & edge : edges) {
            std::tie(from, to, weight) = edge;
            int from_comp_idx = components.findSetByElem(from);
            int   to_comp_idx = components.findSetByElem(to);
            if (from_comp_idx == to_comp_idx) continue;

            components.mergeSets(from_comp_idx, to_comp_idx);

            span_tree.addEdge(from, to, static_cast<float>(weight));
            ++power[from];
            span_tree.addEdge(to, from, static_cast<float>(weight));
            ++power[to];
        }

        std::vector<int> leafs;
        for (const auto & [v, pwr] : power)
            if (pwr == 1) leafs.push_back(v);

        std::map<int, int> max_distances;
        for (const auto leaf : leafs) {
            std::map<int, int> curr_distance;
            span_tree.walkBreadthFirst(leaf, Increment(curr_distance));
            for (const auto v : vertices)
                max_distances[v] = std::max(max_distances[v], curr_distance[v]);
        }

        auto max_dist = std::max_element(max_distances.cbegin(), max_distances.cend(), [](const auto & p1, const auto & p2) {
            return p1.second < p2.second; });

        return std::make_pair(span_tree, max_dist->first);
    }

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
    cv::detail::DisjointSets components;
};


} //namespace utils
