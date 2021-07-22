#ifndef CGCV_ALGORITHMS_H
#define CGCV_ALGORITHMS_H

#include "opencv2/opencv.hpp"

class algorithms
{
   public:
    static void compute_gradient(const cv::Mat &input_image,
                                 cv::Mat &output_image);

    static void compute_binary(const cv::Mat &input_image,
                               const int edge_threshold, cv::Mat &output_image);

    static void crop_image(const cv::Mat &input, const cv::Mat &image_B,
                           const cv::Mat &image_G, const cv::Mat &image_R,
                           const cv::Point2i &offset_B,
                           const cv::Point2i &offset_G, cv::Mat &out_cropped);

    static void combine_images(const cv::Mat &image_B, const cv::Mat &image_G,
                               const cv::Mat &image_R, cv::Mat &output);

    static void translate_img(const cv::Mat &img, const int c_offset,
                              const int r_offset, cv::Mat &output_image);

    static void edge_matching(const std::vector<cv::Mat> &images_BGR,
                              const std::vector<cv::Mat> &img_edges_BGR,
                              const int window_size,
                              std::vector<cv::Mat> &aligned_images_BGR,
                              std::vector<cv::Point2i> &best_offsets_BG);

    static void edge_detection(const std::vector<cv::Mat> &images_BGR,
                               const int edge_threshold,
                               std::vector<cv::Mat> &img_edges_BGR);

    static void bonus(const cv::Mat &image, const int &edge_threshold,
                      const int (&lut)[256], cv::Mat &output);
};

#endif  // CGCV_ALGORITHMS_H
