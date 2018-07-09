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
#include "opencv2/stitching/detail/blenders.hpp"

#include "ba.hpp"
#include "camera.hpp"
#include "utils.hpp"
#include "warper.hpp"


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
    const auto num_of_images = static_cast<int>(images.size());
    if (method == "SIFT") {
        auto finder = cv::xfeatures2d::SIFT::create();
        for (int i = 0; i < num_of_images; ++i) {
            features[i].img_size = images[i].size();
            features[i].img_idx = i;
            finder->detectAndCompute(images[i], cv::Mat(), features[i].keypoints, features[i].descriptors);
        }
    }
    else if (method == "SURF") {
        auto finder = cv::xfeatures2d::SURF::create(300., 3, 4);
        for (int i = 0; i < num_of_images; ++i) {
            features[i].img_size = images[i].size();
            features[i].img_idx = i;
            finder->detectAndCompute(images[i], cv::Mat(), features[i].keypoints, features[i].descriptors);
        }
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
    if (num_of_matches <= 6) return;

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

    auto H = cv::findHomography(src_pts, dst_pts, matches_info.inliers_mask, cv::RANSAC);
    if (std::abs(H.at<double>(2,0)) >= 0.002 or std::abs(H.at<double>(2,1)) >= 0.002)   return;
    // H shouldn't flip vertically, therefore:
    // y coord of H*[0,0,1]^T  <=  y coord of H*[0,1,1]^T  must be true
    if (H.at<double>(1,2) > (H.at<double>(1,1) + H.at<double>(1,2)))    return;
    // same check, but horizontally
    if (H.at<double>(0,2) > (H.at<double>(0,0) + H.at<double>(0,2)))    return;

    matches_info.H = H;
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

    cv::detail::Graph spanning_tree(num_of_images);
    auto graph = utils::AdjacencyMatrix(matches_info, 1.0);    
    auto centers = graph.find_max_span_trees(spanning_tree);

    for (const auto center : centers) {
        auto img_ids = utils::get_vertices_in_component(spanning_tree, center);
        auto num_of_images_ = static_cast<int>(img_ids.size());

        auto features_ = std::vector<cv::detail::ImageFeatures>();
        auto matches_info_ = std::vector<std::vector<cv::detail::MatchesInfo>>();
        auto spanning_tree_ = cv::detail::Graph();
        int center_;
        utils::leave_this_component(img_ids,
            features_, features,
            matches_info_, matches_info,
            spanning_tree_, spanning_tree,
            center_, center);

        auto cameras = camera_params::estimate(features_, matches_info_, spanning_tree_, center_);

        // TODO: BA performance opt.
        auto adjuster = ba::BundleAdjuster(1.0);
        adjuster.adjust(features_, matches_info_, cameras);

        std::vector<cv::Mat> r_matrices;
        for (const auto & cam : cameras)
            r_matrices.push_back(cam.R.clone());
        cv::detail::waveCorrect(r_matrices, cv::detail::WAVE_CORRECT_HORIZ);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = r_matrices[i];

        auto radius = warper::get_radius(cameras);
        auto names = std::vector<std::string>();
        for (int id : img_ids)
            names.push_back(img_names[id]);

        auto warped_imgs = std::vector<cv::UMat>(num_of_images_);
        auto warped_masks = std::vector<cv::UMat>(num_of_images_);
        auto topleft_corners = std::vector<cv::Point>(num_of_images_);
        auto sizes = std::vector<cv::Size>(num_of_images_);
        auto dest_rect = cv::Rect();
        warper::warp(radius, names, cameras, warped_imgs, warped_masks, topleft_corners, sizes, dest_rect);

        auto compensator = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN_BLOCKS);
        compensator->feed(topleft_corners, warped_imgs, warped_masks);
        for (int i = 0; i < num_of_images; ++i)
        	compensator->apply(i, topleft_corners[i], warped_imgs[i], warped_masks[i]);

        auto seam_finder = cv::detail::VoronoiSeamFinder();
        seam_finder.find(warped_imgs, topleft_corners, warped_masks);

        auto area = cv::detail::resultRoi(topleft_corners, sizes).area();
        auto num_of_bands = static_cast<int>(log(sqrt(area)) / log(2.));

        auto blender = cv::detail::MultiBandBlender(false, num_of_bands);
        blender.prepare(dest_rect);
        for (int i = 0; i < num_of_images_; ++i) {
            warped_imgs[i].convertTo(warped_imgs[i], CV_16S);
            blender.feed(warped_imgs[i], warped_masks[i], topleft_corners[i]);
        }

        cv::Mat res, mask;
        blender.blend(res, mask);
        cv::imwrite(std::to_string(center) + ".png", res);
    }
}
