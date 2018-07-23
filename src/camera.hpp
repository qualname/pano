#include <algorithm>
#include <cmath>
#include <vector>

#include "opencv2/opencv.hpp"


namespace camera_params {


struct Estimate_Rotation {
    Estimate_Rotation(const std::vector<std::vector<cv::detail::MatchesInfo>> & matches_info_,
                            std::vector<cv::detail::CameraParams>             & cameras_)
        : cameras(cameras_), matches_info(matches_info_) {}

    cv::Mat K(int camera_idx) const {
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0, 0) = cameras[camera_idx].focal;
        K.at<double>(1, 1) = cameras[camera_idx].focal * cameras[camera_idx].aspect;
        return K;
    }

    void operator()(const cv::detail::GraphEdge & edge) {
        auto K_0 = this->K(edge.from);
        auto K_1 = this->K(edge.to);
        cv::Mat R_01 = K_0.inv() * matches_info[edge.to][edge.from].H * K_1;
        cameras[edge.to].R = cameras[edge.from].R * R_01;
    }

    std::vector<cv::detail::CameraParams> & cameras;
    std::vector<std::vector<cv::detail::MatchesInfo>> matches_info;
};

double focal_from_homography(const cv::Mat & H)
{
    if (H.empty()) return -1.0;

    const auto h = H.ptr<double>();
    double f0_squared, f1_squared;

    double denom_1, denom_2;
    double val_1, val_2;

    denom_1 = h[6] * h[7];
    denom_2 = (h[7] - h[6]) * (h[7] + h[6]);
    val_1 = - (h[0] * h[1] + h[3] * h[4]) / denom_1;
    val_2 = (h[0]*h[0] + h[3]*h[3] - h[1]*h[1] - h[4]*h[4]) / denom_2;

    if (val_1 < val_2) std::swap(val_1, val_2);

    if      (val_2 > 0) f1_squared = std::abs(denom_1) > std::abs(denom_2) ? val_1
                                                                           : val_2;
    else if (val_1 > 0) f1_squared = val_1;
    else                return -1.0;

    denom_1 = h[0] * h[3] + h[1] * h[4];
    denom_2 = h[0]*h[0] + h[1]*h[1] - h[3]*h[3] - h[4]*h[4];
    val_1 = - (h[2] * h[5]) / denom_1;
    val_2 = (h[5]*h[5] - h[2]*h[2]) / denom_2;

    if (val_1 < val_2) std::swap(val_1, val_2);
    if      (val_2 > 0) f0_squared = std::abs(denom_1) > std::abs(denom_2) ? val_1
                                                                   : val_2;
    else if (val_1 > 0) f0_squared = val_1;
    else                return -1.0;

    return std::sqrt(std::sqrt(f0_squared) * std::sqrt(f1_squared));
}

double estimate_focal(const std::vector<cv::detail::ImageFeatures>            & features,
                      const std::vector<std::vector<cv::detail::MatchesInfo>> & matches_info)
{
    auto focals = std::vector<double>();
    for (const auto & row : matches_info) {
        for (const auto & m : row) {
            auto focal = focal_from_homography(m.H);
            if (focal > 0)
                focals.push_back(focal);
        }
    }

    std::sort(focals.begin(), focals.end());

    const auto size = focals.size();
    if      (size == 0)     return -1.0;
    else if (size % 2 == 0) return (focals[size / 2 - 1] + focals[size / 2]) / 2;
    else                    return  focals[size / 2];
}

std::vector<cv::detail::CameraParams>
estimate(const std::vector<cv::detail::ImageFeatures>            & features,
         const std::vector<std::vector<cv::detail::MatchesInfo>> & matches_info,
         const cv::detail::Graph                                 & spanning_tree,
         const int                                                 center)
{
    const auto num_of_images = static_cast<int>(features.size());
    auto cameras = std::vector<cv::detail::CameraParams>(num_of_images, cv::detail::CameraParams());

    const auto focal = estimate_focal(features, matches_info);
    for (auto & cam : cameras)
        cam.focal = focal;

    spanning_tree.walkBreadthFirst(center, Estimate_Rotation(matches_info, cameras));

    for (int i = 0; i < num_of_images; ++i) {
        cameras[i].ppx += 0.5 * features[i].img_size.width;
        cameras[i].ppy += 0.5 * features[i].img_size.height;
        cameras[i].R.convertTo(cameras[i].R, CV_32F);
    }

    return cameras;
}


} //namespace camera_params
