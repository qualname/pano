#include <cmath>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

namespace warper {


float get_radius(const std::vector<cv::detail::CameraParams> & cams)
{
    auto num = cams.size();

    std::vector<double> f;
    for (const auto & cam : cams)
        f.push_back(cam.focal);
    std::sort(f.begin(), f.end());

    return static_cast<float>(f[num / 2]);
}

void warp(const float radius,
          const std::vector<std::string>              & img_names,
          const std::vector<cv::detail::CameraParams> & cams,
                std::vector<cv::UMat>                 & warped,
                cv::Rect                              & dest)
{
    auto num_of_imgs = static_cast<int>(img_names.size());
    auto topleft = std::vector<cv::Point>(num_of_imgs);
    auto img_sizes = std::vector<cv::Size>(num_of_imgs);

    auto warper = cv::SphericalWarper().create(radius);
    for (int i = 0; i < num_of_imgs; ++i) {
        cv::Mat_<float> K;
        cams[i].K().convertTo(K, CV_32F);

        auto img = cv::imread(img_names[i]);

        cv::UMat mapx, mapy;
        auto roi = warper->buildMaps(img.size(), K, cams[i].R, mapx, mapy);
        warped[i].create(roi.height + 1, roi.width + 1, img.type());
        cv::remap(img, warped[i], mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

        topleft[i] = roi.tl();
        img_sizes[i] = warped[i].size();
    }
    dest = cv::detail::resultRoi(topleft, img_sizes);

    delete warper;
}


} // namespace warper
