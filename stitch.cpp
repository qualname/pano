#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/stitching/detail/matchers.hpp"

#include "camera.hpp"
#include "utils.hpp"

//double SCALE = 0.868757;  // TODO
double SCALE = 1.0;


std::vector<std::string> parse_args(int argc, char * argv[])
{
    std::vector<std::string> img_names;
    for (int i = 1; i < argc; ++i){
        img_names.push_back(argv[i]);
    }

    return img_names;
}

void read_images(const std::vector<std::string> & image_names,
                       std::vector<cv::Mat>     & images,
                       std::vector<cv::Size>    & image_sizes)
{
    cv::Mat img, resized_img;
    for (const auto & name : image_names) {
        img = cv::imread(name, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Couldn't open: " << name << "\n";
            std::exit(-1);
        }

        if (SCALE == 1.0)
            resized_img = img;
        else
            cv::resize(img, resized_img, cv::Size(), SCALE, SCALE, cv::INTER_LINEAR_EXACT);

        image_sizes.push_back(resized_img.size());
        images.push_back(resized_img.clone());
    }
}

void find_feature_points(const std::vector<cv::Mat>                   & images,
                               std::vector<cv::detail::ImageFeatures> & features,
                               std::string                              method="SIFT")
{
    if (method == "SIFT") {
        const auto num_of_images = static_cast<int>(images.size());
        auto finder = cv::xfeatures2d::SIFT::create();
        for (int i = 0; i < num_of_images; ++i) {
            finder->detectAndCompute(images[i], cv::Mat(), features[i].keypoints, features[i].descriptors);
        }
    }
    else if (method == "SURF") {
        auto finder = cv::detail::SurfFeaturesFinder();
        finder(images, features);
        finder.collectGarbage();
    }
}

void match_features(const cv::detail::ImageFeatures & features1,
                    const cv::detail::ImageFeatures & features2,
                          cv::detail::MatchesInfo   & matches_info)
{
    auto matcher = cv::FlannBasedMatcher();

    std::vector<std::vector<cv::DMatch>> matches;
    matcher.knnMatch(features1.descriptors, features2.descriptors, matches, 2);

    for (const auto & m : matches) {
        if (m.size() < 2) continue;
        if (m[0].distance < 0.75f * m[1].distance) {
            matches_info.matches.push_back(m[0]);
        }
    }
}

void get_homography(const cv::detail::ImageFeatures & features1,
                    const cv::detail::ImageFeatures & features2,
                          cv::detail::MatchesInfo   & matches_info)
{
    const auto num_of_matches = static_cast<int>(matches_info.matches.size());
    assert (num_of_matches > 6);

    auto src_pts = cv::Mat(1, num_of_matches, CV_32FC2);
    auto dst_pts = cv::Mat(1, num_of_matches, CV_32FC2);
    for (int i = 0; i < num_of_matches; ++i) {
        const auto & m = matches_info.matches[i];

        auto p = features1.keypoints[m.queryIdx].pt;
        p.x -= features1.img_size.width  * 0.5f;
        p.y -= features1.img_size.height * 0.5f;
        src_pts.at<cv::Point2f>(0, i) = p;

        p = features2.keypoints[m.trainIdx].pt;
        p.x -= features2.img_size.width  * 0.5f;
        p.y -= features2.img_size.height * 0.5f;
        dst_pts.at<cv::Point2f>(0, i) = p;
    }

    matches_info.H = cv::findHomography(src_pts, dst_pts, matches_info.inliers_mask, cv::RANSAC);

    matches_info.num_inliers = 0;
    for (auto inlier : matches_info.inliers_mask) {
        if (inlier)
            ++matches_info.num_inliers;
    }

    matches_info.confidence = matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());
}

void match_feature_points(const std::vector<cv::detail::ImageFeatures>            & features,
                                std::vector<std::vector<cv::detail::MatchesInfo>> & matches_info)
{
    const auto num_of_images = static_cast<int>(features.size());
    for (int i = 0; i < num_of_images - 1; ++i) {
        for (int j = i + 1; j < num_of_images; ++j) {
            match_features(features[i], features[j], matches_info[i][j]);
            get_homography(features[i], features[j], matches_info[i][j]);
            matches_info[i][j].src_img_idx = i;
            matches_info[i][j].dst_img_idx = j;


            matches_info[j][i] = matches_info[i][j];

            const auto & H = matches_info[i][j].H;
            if (not H.empty())
                matches_info[j][i].H = H.inv();

            std::swap(matches_info[j][i].src_img_idx,
                      matches_info[j][i].dst_img_idx);

            for (auto & m : matches_info[j][i].matches) {
                std::swap(m.queryIdx, m.trainIdx);
            }
        }
    }
}


int main(int argc, char * argv[])
{
    auto img_names = parse_args(argc, argv);
    auto num_of_images = static_cast<int>(img_names.size());

    cv::Mat image;
    std::vector<cv::Mat>  images;
    std::vector<cv::Size> image_sizes;
    read_images(img_names, images, image_sizes);

    auto features = std::vector<cv::detail::ImageFeatures>(num_of_images);
    find_feature_points(images, features, "SIFT");

    auto matches_info = std::vector<std::vector<cv::detail::MatchesInfo>>(
        num_of_images, std::vector<cv::detail::MatchesInfo>(num_of_images)
    );
    match_feature_points(features, matches_info);

    auto graph = utils::AdjacencyMatrix(matches_info, 1.0);
    auto components = graph.find_components();

    std::vector<std::pair<cv::detail::Graph, int>> trees;
    for (const auto & comp : components)
        trees.push_back(graph.find_max_span_tree(comp));

    auto cameras = std::vector<std::vector<cv::detail::CameraParams>>();
    for (const auto & [tree, center] : trees)
        cameras.push_back(camera_params::estimate(features, matches_info, tree, center));
}
