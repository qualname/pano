#include <string>
#include <vector>


std::vector<std::string> parse_args(int argc, char * argv[])
{
    std::vector<std::string> img_names;
    for (int i = 1; i < argc; ++i){
        img_names.push_back(argv[i]);
    }

    return img_names;
}


int main(int argc, char * argv[])
{
    auto img_names = parse_args(argc, argv);
    auto num_of_images = static_cast<int>(img_names.size());
}
