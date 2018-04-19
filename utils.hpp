#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/util.hpp"


namespace utils {


struct UNION_WHERE {
    UNION_WHERE(int num_of_vertices)
    {
        for (int i = 0; i < num_of_vertices; ++i)
            _vertices.emplace(i, -1);
    }

    int find_set(int vertex)
    {
        int parent = _vertices.at(vertex);

        while (parent != -1) {
            vertex = parent;
            parent = _vertices.at(vertex);
        }
        return vertex;
    }

    void union_(int from, int to)
    {
        from = find_set(from);
        to   = find_set(to);
        _vertices[to] = from;
    }

    std::map<int, std::vector<int>> get_comps()
    {
        std::map<int, std::vector<int>> comps;
        for (const auto & v : _vertices) {
            auto parent = find_set(v.first);
            comps[parent].push_back(v.first);
        }
        return comps;
    }

private:
    std::map<int, int> _vertices;
};


struct CmpWeightDesc {
    bool operator()(const std::tuple<int, int, double> & t1,
                    const std::tuple<int, int, double> & t2)
    {
        return std::get<2>(t1) >= std::get<2>(t2);
    }
};


struct AccumulateOutEdges {
    AccumulateOutEdges(int vertex, std::vector<cv::detail::GraphEdge> * edges)
    : _vertex(vertex),
      _edges(edges)
    {}

    void operator() (const cv::detail::GraphEdge & edge) {
        if (edge.from == _vertex) _edges->push_back(edge);
    }

    int _vertex;
    std::vector<cv::detail::GraphEdge> * _edges;
};


std::vector<cv::detail::GraphEdge> adjacent(int start, const cv::detail::Graph & span_tree)
{
    std::vector<cv::detail::GraphEdge> edges;
    auto acc = AccumulateOutEdges(start, &edges);
    span_tree.forEach(acc);
    return edges;
}


int get_depth(int start, const cv::detail::Graph & span_tree, std::vector<bool> & discovered)
{
    int depth = 0;

    discovered[start] = true;
    for (const auto & edge : adjacent(start, span_tree)) {
        if (not discovered[edge.to]) {
            auto d = 1 + get_depth(edge.to, span_tree, discovered);
            depth = std::max(depth, d);
        }
    }
    return depth;
}


class AdjacencyMatrix {
public:
    AdjacencyMatrix(const std::vector<std::vector<cv::detail::MatchesInfo>> & matrix, double threshold)
    : _max_vertex_idx(static_cast<int>(matrix.size()))
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

    std::vector<int> find_max_span_trees(cv::detail::Graph & span_tree)
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
                span_tree.addEdge(from, to, static_cast<float>(weight));
                span_tree.addEdge(to, from, static_cast<float>(weight));
                set.union_(from, to);
            }
        }

        std::vector<int> centers;
        _components = set.get_comps();        
        for (const auto & [root, vertices] : _components) {
            int curr_depth, min_depth = static_cast<int>(vertices.size());
            int min_vertex;
            for (int v : vertices) {
                auto seen = std::vector<bool>(_max_vertex_idx, false);
                curr_depth = get_depth(v, span_tree, seen);

                if (curr_depth < min_depth) {
                    min_depth = curr_depth;
                    min_vertex = v;
                }
            }
            centers.push_back(min_vertex);
        }

        return centers;
    }

private:
    int _max_vertex_idx;
    std::map<int, std::vector<int>> _components;
    std::vector<std::vector<double>> adj_matrix;
};


struct Acc {
    Acc(std::vector<int> * vertices_, int v_)
    : vertices(vertices_)
    {
        vertices->push_back(v_);
    }

    void operator() (const cv::detail::GraphEdge & edge) {
        vertices->push_back(edge.to);
    }

    std::vector<int> * vertices;
};


std::vector<int> get_vertices_in_component(const cv::detail::Graph & span_tree, const int center)
{
    std::vector<int> vertices;
    auto acc = Acc(&vertices, center);
    span_tree.walkBreadthFirst(center, acc);
    std::sort(vertices.begin(), vertices.end());
    return vertices;
}


struct Copy {
    Copy(cv::detail::Graph * g_, const std::vector<int> & ind_)
    : g(g_)
    {
        for (int i = 0; i < static_cast<int>(ind_.size()); ++i)
            map[ind_[i]] = i;
    }

    void operator() (const cv::detail::GraphEdge & edge) {
        if (map.find(edge.from) != map.end())
            g->addEdge(map[edge.from], map[edge.to], edge.weight);
    }

    cv::detail::Graph * g;
    std::map<int, int> map;
};

void leave_this_component(const std::vector<int> & img_ids,
    std::vector<cv::detail::ImageFeatures> & features_, const std::vector<cv::detail::ImageFeatures> & features,
    std::vector<std::vector<cv::detail::MatchesInfo>> & matches_, const std::vector<std::vector<cv::detail::MatchesInfo>> & matches,
    cv::detail::Graph & span_tree_, const cv::detail::Graph & span_tree,
    int & center_, const int center)
{
    auto size = static_cast<int>(img_ids.size());
    
    matches_.resize(size);
    for (int i = 0; i < size; ++i) {
        int i_ = img_ids[i];

        features_.push_back(features[i_]);
        matches_[i].resize(size);
        for (int j = 0; j < size; ++j) {
            int j_ = img_ids[j];
            matches_[i][j] = matches[i_][j_];
        }
    }

    span_tree_.create(size);
    Copy copy(&span_tree_, img_ids);
    span_tree.forEach(copy);

    for (int i = 0; i < size; ++i) {
        if (img_ids[i] == center) {
            center_ = i;
            break;
        }
    }
}


} //namespace utils
