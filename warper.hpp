#include <vector>

#include "opencv2/opencv.hpp"

namespace warper {


class SphericalMap;


float get_radius(const std::vector<cv::detail::CameraParams> & cams)
{
    auto num = cams.size();

    std::vector<double> f;
    for (const auto & cam : cams)
        f.push_back(cam.focal);
    std::sort(f.begin(), f.end());

    return static_cast<float>(f[num / 2]);
}


} // namespace warper
