#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include "algorithms.h"
#include "opencv2/opencv.hpp"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

#define RST "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define FRED(x) KRED x RST
#define FGRN(x) KGRN x RST
#define FYEL(x) KYEL x RST
#define FBLU(x) KBLU x RST
#define FMAG(x) KMAG x RST
#define FCYN(x) KCYN x RST
#define FWHT(x) KWHT x RST

#define BOLD(x) "\x1B[1m" x RST
#define UNDL(x) "\x1B[4m" x RST

//===============================================================================
// make_directory()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void make_directory(const char* path)
{
#if defined(_WIN32)
    _mkdir(path);
#else
    mkdir(path, 0777);
#endif
}

//===============================================================================
// get_out_filenames()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
std::vector<std::string> get_out_filenames(const rapidjson::Value& config,
                                           int number_width, bool zero_filled,
                                           const std::string out_filename_array,
                                           const std::string out_full_path,
                                           const std::string out_filetype)
{
    const rapidjson::Value& a = config[out_filename_array.c_str()];
    std::vector<std::string> members;
    for (rapidjson::SizeType i = 0; i < a.Size(); i++)
    {
        if (a[i].IsString())
        {
            std::string out_filename = a[i].GetString();
            std::string digit_string = std::to_string(i + 1);
            std::string zero_digit_string =
                zero_filled
                    ? std::string(number_width - digit_string.length(), '0') +
                          digit_string
                    : digit_string;
            members.emplace_back(out_full_path + zero_digit_string + "_" +
                                 out_filename + out_filetype);
        }
    }
    return members;
}

//===============================================================================
// save_image()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void save_image(std::vector<std::string> out_filenames, cv::Mat image)
{
    static int out_file_index = 0;
    cv::imwrite(out_filenames.at(out_file_index), image);
    std::cout << "saving image: " << out_filenames.at(out_file_index)
              << std::endl;
    ++out_file_index;
}

//===============================================================================
// run()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void run(std::vector<cv::Mat> input_images_BGR,
         std::vector<std::string> out_filenames, std::string out_filetype,
         int edge_threshold, int matching_window_size, int c_offset,
         int r_offset)
{
    // Edge Images
    cv::Mat img_b_edge = cv::Mat::zeros(input_images_BGR.at(0).size(), CV_8UC1);
    cv::Mat img_g_edge = cv::Mat::zeros(input_images_BGR.at(1).size(), CV_8UC1);
    cv::Mat img_r_edge = cv::Mat::zeros(input_images_BGR.at(2).size(), CV_8UC1);
    std::vector<cv::Mat> img_edges_BGR;
    img_edges_BGR.push_back(img_b_edge);
    img_edges_BGR.push_back(img_g_edge);
    img_edges_BGR.push_back(img_r_edge);

    // Aligned images
    cv::Mat img_b_aligned =
        cv::Mat::zeros(input_images_BGR.at(0).size(), CV_8UC1);
    cv::Mat img_g_aligned =
        cv::Mat::zeros(input_images_BGR.at(1).size(), CV_8UC1);
    std::vector<cv::Mat> img_aligned_BG;
    img_aligned_BG.push_back(img_b_aligned);
    img_aligned_BG.push_back(img_g_aligned);

    std::vector<cv::Point2i> best_offsets_BG(2);

    // Output image
    cv::Mat out = cv::Mat::zeros(input_images_BGR.at(2).size(), CV_8UC3);

    // Cropped output image
    cv::Mat out_cropped =
        cv::Mat::zeros(input_images_BGR.at(2).size(), CV_8UC3);

    //=============================================================================
    // Gradient image
    //=============================================================================
    std::cout << "Step 1 - calculating gradient image... " << std::endl;
    cv::Mat gradient_image =
        cv::Mat::zeros(input_images_BGR.at(2).size(), CV_8UC1);
    algorithms::compute_gradient(input_images_BGR.at(2), gradient_image);
    save_image(out_filenames, gradient_image);

    //=============================================================================
    // Binary image
    //=============================================================================
    std::cout << "Step 2 - calculating binary image... " << std::endl;
    cv::Mat binary_image =
        cv::Mat::zeros(input_images_BGR.at(2).size(), CV_8UC1);
    algorithms::compute_binary(gradient_image, edge_threshold, binary_image);
    save_image(out_filenames, binary_image);

    //=============================================================================
    // Edge detection
    //=============================================================================
    std::cout << "Step 3 - calculating edge images... " << std::endl;
    algorithms::edge_detection(input_images_BGR, edge_threshold, img_edges_BGR);

    save_image(out_filenames, img_edges_BGR.at(0));
    save_image(out_filenames, img_edges_BGR.at(1));
    save_image(out_filenames, img_edges_BGR.at(2));

    //=============================================================================
    // Translate image
    //=============================================================================
    std::cout << "Step 4 - checking translated image... " << std::endl;
    cv::Mat translate_image =
        cv::Mat::zeros(input_images_BGR.at(2).size(), CV_8UC1);
    algorithms::translate_img(input_images_BGR.at(2), c_offset, r_offset,
                              translate_image);
    save_image(out_filenames, translate_image);

    //=============================================================================
    // Edge matching
    //=============================================================================
    std::cout << "Step 5 - determining best matching offset and transforming "
                 "images... "
              << std::endl;

    algorithms::edge_matching(input_images_BGR, img_edges_BGR,
                              matching_window_size, img_aligned_BG,
                              best_offsets_BG);

    save_image(out_filenames, img_aligned_BG.at(0));
    save_image(out_filenames, img_aligned_BG.at(1));

    //=============================================================================
    // Combine images
    //=============================================================================
    std::cout << "Step 6 - combining images... " << std::endl;

    algorithms::combine_images(img_aligned_BG.at(0), img_aligned_BG.at(1),
                               input_images_BGR.at(2), out);

    save_image(out_filenames, out);

    //=============================================================================
    // Crop image
    //=============================================================================
    std::cout << "Step 7 - cropping output image... " << std::endl;

    algorithms::crop_image(out, img_aligned_BG.at(0), img_aligned_BG.at(1),
                           input_images_BGR.at(2), best_offsets_BG.at(0),
                           best_offsets_BG.at(1), out_cropped);

    save_image(out_filenames, out_cropped);

    //=============================================================================
    // Bonus
    //=============================================================================
    std::cout << "Starting BONUS Task..." << std::endl;

    cv::Mat output_cartoon = cv::Mat::zeros(out_cropped.size(), CV_8UC3);
    int lut[256];
    for (int i = 0; i < sizeof(lut) / sizeof(lut[0]); i++)
    {
        lut[i] = floor(i / 32) * 32 + 16;
    }

    algorithms::bonus(out_cropped, 55, lut, output_cartoon);
    save_image(out_filenames, output_cartoon);
}

//===============================================================================
// execute_testcase()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void execute_testcase(const rapidjson::Value& config_data)
{
    //=============================================================================
    // Parse input data
    //=============================================================================
    std::string name = config_data["name"].GetString();
    size_t edge_threshold = config_data["edge_threshold"].GetUint();
    size_t matching_window_size = config_data["matching_window_size"].GetUint();
    size_t c_offset = config_data["c_offset"].GetInt();
    size_t r_offset = config_data["r_offset"].GetInt();

    //=============================================================================
    // Load input images
    //=============================================================================
    std::string input_directory = "./data/" + name + "/";

    std::cout << BOLD(FGRN("[INFO]")) << " Input path: " << input_directory
              << std::endl;

    std::vector<cv::Mat> input_images_BGR;
    cv::Mat img_r =
        cv::imread(input_directory + "01.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_g =
        cv::imread(input_directory + "02.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_b =
        cv::imread(input_directory + "03.png", cv::IMREAD_GRAYSCALE);
    input_images_BGR.push_back(img_b);
    input_images_BGR.push_back(img_g);
    input_images_BGR.push_back(img_r);

    for (int i = 0; i < input_images_BGR.size(); i++)
    {
        std::string filename = input_directory + std::to_string(i) + ".png";
        if (!input_images_BGR.at(i).data)
        {
            std::cout << BOLD(FRED("[ERROR]")) << " Could not load image ("
                      << filename << ")" << std::endl;
            throw std::runtime_error("Could not load file");
        }
    }

    //=============================================================================
    // Create output directory
    //=============================================================================
    std::string output_directory = "./output/" + name + "/";

    std::cout << BOLD(FGRN("[INFO]")) << " Output path: " << input_directory
              << std::endl;

    make_directory("./output/");
    make_directory(output_directory.c_str());

    //=============================================================================
    // Get output filenames
    //=============================================================================
    std::vector<std::string> out_filenames =
        get_out_filenames(config_data, 2, true, std::string("out_filenames"),
                          output_directory, std::string(".png"));

    //=============================================================================
    // Starting default task
    //=============================================================================
    std::cout << "Starting MAIN Task..." << std::endl;
    run(input_images_BGR, out_filenames, std::string(".png"), edge_threshold,
        matching_window_size, c_offset, r_offset);
}

//===============================================================================
// main()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
int main(int argc, char* argv[])
{
    std::cout << "CV/task1 framework version 1.0"
              << std::endl;  // DO NOT REMOVE THIS LINE!!!
    std::cout << "===================================" << std::endl;
    std::cout << "               CV Task 1           " << std::endl;
    std::cout << "===================================" << std::endl;

    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <config-file>" << std::endl;
        return 1;
    }

    std::ifstream fs(argv[1]);
    if (!fs)
    {
        std::cout << "Error: Failed to open file " << argv[1] << std::endl;
        return 2;
    }
    std::stringstream buffer;
    buffer << fs.rdbuf();

    rapidjson::Document doc;
    rapidjson::ParseResult check;
    check = doc.Parse<0>(buffer.str().c_str());

    if (check)
    {
        if (doc.HasMember("testcases"))
        {
            rapidjson::Value& testcases = doc["testcases"];
            for (rapidjson::SizeType i = 0; i < testcases.Size(); i++)
            {
                rapidjson::Value& testcase = testcases[i];
                try
                {
                    execute_testcase(testcase);
                }
                catch (const std::exception& e)
                {
                    std::cout << e.what() << std::endl;
                    std::cout << BOLD(FRED("[ERROR]"))
                              << " Program exited with errors!" << std::endl;
                    return -1;
                }
            }
        }
        std::cout << "Program exited normally!" << std::endl;
    }
    else
    {
        std::cout << "Error: Failed to parse file " << argv[1] << ":"
                  << check.Offset() << std::endl;
        return 3;
    }
    return 0;
}
