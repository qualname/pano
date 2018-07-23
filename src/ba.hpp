#include <vector>

#include "opencv2/opencv.hpp"


namespace ba {

class BundleAdjuster {
public:
    BundleAdjuster(double conf) {
        _impl.setConfThresh(static_cast<float>(conf));

        cv::Mat_<uchar> mask = cv::Mat::zeros(3, 3, CV_8U);
        mask(0,0) = mask(0,1) = mask(0,2) = 1;
        mask(1,1) = mask(1,2) = 1;

        _impl.setRefinementMask(mask);
    }

    void adjust(const std::vector<cv::detail::ImageFeatures>            & features,
                const std::vector<std::vector<cv::detail::MatchesInfo>> & matches,
                      std::vector<cv::detail::CameraParams>             & cameras)
    {
        auto N = static_cast<int>(matches.size());
        std::vector<cv::detail::MatchesInfo> matches_(N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                matches_[i * N + j] = matches[i][j];

        _impl(features, matches_, cameras);
    }

private:
    cv::detail::BundleAdjusterRay _impl = cv::detail::BundleAdjusterRay();
};

} //namespace ba
