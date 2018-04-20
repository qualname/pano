#include <cmath>
#include <vector>

#include "opencv2/opencv.hpp"

namespace warper {


class SphericalMap {
public:
    SphericalMap(float s_) : s(s_) {} 

    void map_coord(const h_point & p)
    {
        float x_ = s * std::atan2(p.x, p.z);
        float y_ = s * std::atan2(p.y, std::hypot(p.x, p.z));
    }

    void inv_map_coord(const point & p_)
    {
        float x = std::tan(p_.x / s);
        float y = std::tan(p_.y / s) / std::cos(p_.x / s);
        float z = 1.f;
    }

private:
    float s;
};


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
