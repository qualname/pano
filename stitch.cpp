#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/mat.hpp"


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


int main(int argc, char * argv[])
{
    auto img_names = parse_args(argc, argv);
    auto num_of_images = static_cast<int>(img_names.size());

    cv::Mat image;
    std::vector<cv::Mat>  images;
    std::vector<cv::Size> image_sizes;
    read_images(img_names, images, image_sizes);
}
