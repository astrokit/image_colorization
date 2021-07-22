#include "algorithms.h"


//===============================================================================
// compute_gradient()
//-------------------------------------------------------------------------------
// TODO: Calculate the 1st Sobel derivative once in x and once in y direction
//       and combine these two.
//
// parameters:
//  - input_image: the image for the gradient calculation
//  - output_image: output matrix for the gradient image
// return: void
//===============================================================================
void algorithms::compute_gradient(const cv::Mat &input_image,
                                  cv::Mat &output_image)
{
    // Compute first derivative in x and y direction
    cv::Mat grad_x, grad_y;
    cv::Sobel(input_image, grad_x, CV_32F, 1, 0);
    cv::Sobel(input_image, grad_y, CV_32F, 0, 1);

    // Combine the gradient images -> sqrt(pow(grad_x, 2) + pow(grad_y, 2))
    cv::Mat grad_x_square, grad_y_square, grad_square_sum;
    cv::pow(grad_x, 2, grad_x_square);
    cv::pow(grad_y, 2, grad_y_square);

    cv::add(grad_x_square, grad_y_square, grad_square_sum);
    cv::sqrt(grad_square_sum, output_image);
}

//===============================================================================
// compute_binary()
//-------------------------------------------------------------------------------
// TODO: 1) Normalize every gradient image and convert the results to CV_8UC1.
//       2) Threshold the retrieved (normalized) gradient image using the
//          parameter "edge_threshold".
//
// parameters:
//  - input_image: the image for the binary image calculation
//  - edge_threshold: intensities above this value are edges
//  - output_image: output matrix for the binary image
// return: void
//===============================================================================
void algorithms::compute_binary(const cv::Mat &input_image,
                                const int edge_threshold, cv::Mat &output_image)
{
    // Find min and max pixel values from the gradient image
    double p_min, p_max;
    cv::minMaxLoc(input_image, &p_min, &p_max);

    // Convert to 8-bit grayscale image and scale all pixels to the interval [0, 255]
    cv::Mat binary_image;
    input_image -= p_min; // x - p_min
    double scale_factor = 255.0 / (p_max - p_min);
    input_image.convertTo(binary_image, CV_8UC1, scale_factor);

    // Binary threshold: Set all pixel values > edge_threshold to 255, otherwise to 0
    cv::threshold(binary_image, output_image, edge_threshold, 255.0, cv::THRESH_BINARY);
}

//===============================================================================
// edge_detection()
//-------------------------------------------------------------------------------
// TODO: 1) Use compute_gradient for the gradient calculation of each channel
//       2) Use compute_binary for the binary image calculation of each channel
//       3) Save the results in the img_edge_BGR.
//       Note: Each channel has a different image size.
//
// parameters:
//  - images_BGR: vector of input images in order blue, green, red
//  - edge_threshold: intensities above this value are edges
//  - img_edges_BGR: vector for output images (storage order: blue, green, red)
//                  (vector has already the proper size of 3 matrices)
// return: void
//===============================================================================
void algorithms::edge_detection(const std::vector<cv::Mat> &images_BGR,
                                const int edge_threshold,
                                std::vector<cv::Mat> &img_edges_BGR)
{
    for (size_t color_index = 0; color_index < 3; color_index++)
    {
        const auto& color_channel = images_BGR.at(color_index);
        cv::Mat gradient_image = cv::Mat::zeros(color_channel.size(), CV_8UC1);

        compute_gradient(color_channel, gradient_image);
        compute_binary(gradient_image, edge_threshold, img_edges_BGR.at(color_index));
    }
}

//===============================================================================
// translate_img()
//-------------------------------------------------------------------------------
// TODO: 1) Translate the input image by the given offsets.
//       2) This leads to pixels which are not in the image range, set them to 0.
//       Note: A positive offset should lead to a shift to the right/bottom.
//
// parameters:
//  - img: the image to be translated
//  - c_offset: the offset for the columns
//  - r_offset: the offset for the rows
//  - output_image: Return the translated image here.
// return: void
//===============================================================================
void algorithms::translate_img(const cv::Mat &img, const int c_offset,
                               const int r_offset, cv::Mat &output_image)
{
    // Make sure each pixel of output_image gets a pixel value assigned
    for (int x = 0; x < output_image.rows; x++)
    {
        for (int y = 0; y < output_image.cols; y++)
        {
            uchar pixel = 0;

            // Find the corresponding offset pixel in img
            int img_x_offset = x - r_offset;
            int img_y_offset = y - c_offset;

            // Ensure that we don't access a pixel in img that cannot exist, e.g., img.at<uchar>(-5, 0)
            if (0 <= img_x_offset && img_x_offset < img.rows &&
                0 <= img_y_offset && img_y_offset < img.cols)
            {
                pixel = img.at<uchar>(img_x_offset, img_y_offset);
            }

            output_image.at<uchar>(x, y) = pixel;
        }
    }
}

//===============================================================================
// edge_matching()
//-------------------------------------------------------------------------------
// TODO: 1) Determine the best matching offset for the channels G and B. To
//          accomplish this the images are transformed (shifted in the x and
//          y direction) and a score for every possible transformation is
//          calculated by checking how many edge pixels in the reference channel
//          (R) lie on edge pixels in the transformed channels (G, B).
//       2) After calculating the best offset, transform the channels according
//          to this offset and save them to vector aligned_images_BG.
//       Note: The expected output images are of the same size as their
//             untransformed original images.
//
// parameters:
//  - input_images_BGR: vector of input images in order blue, green, red
//  - img_edges_BGR: vector of edge images in order blue, green, red
//  - match_window_size: size of pixel neighborhood in which edges should be
//                       searched
//  - aligned_images_BG: output vector for aligned channel images
//                       (storage order: blue, green)
//                       (vector has already the proper size of 2 matrices)
//  - best_offsets_BG: output vector for the best offsets
//                     (storage order: blue, green)
//                     (vector has already the proper size of 2 points)
// return: void
//===============================================================================
void algorithms::edge_matching(const std::vector<cv::Mat> &input_images_BGR,
                               const std::vector<cv::Mat> &img_edges_BGR,
                               const int match_window_size,
                               std::vector<cv::Mat> &aligned_images_BG,
                               std::vector<cv::Point2i> &best_offsets_BG)
{
    int r = match_window_size / 2;
    const auto& img_edge_r = img_edges_BGR.at(2); // = E

    // Align the edges of the blue and green channels with the edges of the red channel
    for (size_t color_index = 0; color_index < 2; color_index++) // 0 => blue, 1 => green
    {
        const auto& img_edge = img_edges_BGR.at(color_index);
        cv::Mat img_edge_translated = cv::Mat::zeros(img_edge.size(), CV_8UC1);

        // Translate the edge image within the interval r_min and r_max => [-r, r]
        int score_max = 0, i_max = 0, j_max = 0;
        for (int i = -r; i <= r; i++)
        {
            for (int j = -r; j <= r; j++)
            {
                translate_img(img_edge, i, j, img_edge_translated); // = E'

                // Count how many edge pixels (= value 255) are at the same coordinates in E and E'
                int score_i_j = 0;
                for (int x = 0; x < img_edge_r.rows; x++)
                {
                    for (int y = 0; y < img_edge_r.cols; y++)
                    {
                        if (img_edge_r.at<uchar>(x, y) == 255 && img_edge_translated.at<uchar>(x, y) == 255)
                        {
                            score_i_j++;
                        }
                    }
                }

                // If the current translation (i,j) maximizes the score function => store it as temporary optimum
                if (score_i_j > score_max)
                {
                    score_max = score_i_j;
                    i_max = i;
                    j_max = j;
                }
            }
        }

        // Translate the color channel with the found optimum and store it
        const auto& color_channel = input_images_BGR.at(color_index);
        translate_img(color_channel, i_max, j_max, aligned_images_BG.at(color_index));
        best_offsets_BG.at(color_index) = cv::Point2i(i_max, j_max);
    }
}

//===============================================================================
// combine_images()
//-------------------------------------------------------------------------------
// TODO: Combine the three image channels into one single image. Mind the pixel
//       format!
//       Note: The expected image has the same dimensions as the reference
//             channel (R).
//
// parameters:
//  - image_B: blue input channel
//  - image_G: green input channel
//  - image_R: red input channel
//  - output: output matrix for combined image
//            (matrix has the size of the red channel)
// return: void
//===============================================================================
void algorithms::combine_images(const cv::Mat &image_B, const cv::Mat &image_G,
                                const cv::Mat &image_R, cv::Mat &output)
{
    for (int x = 0; x < output.rows; x++)
    {
        for (int y = 0; y < output.cols; y++)
        {
            // If x or y does not exist in one of the color channels => set to 0
            uchar b = (x < image_B.rows && y < image_B.cols) ? image_B.at<uchar>(x, y) : 0;
            uchar g = (x < image_G.rows && y < image_G.cols) ? image_G.at<uchar>(x, y) : 0;
            uchar r = (x < image_R.rows && y < image_R.cols) ? image_R.at<uchar>(x, y) : 0;

            output.at<cv::Vec3b>(x, y) = cv::Vec3b(b, g, r);
        }
    }
}

//===============================================================================
// crop_image()
//-------------------------------------------------------------------------------
// TODO: Crop the above generated image s.t. only pixels defined in every
//       channel remain in the resulting cropped output image.
//
// parameters:
//  - input: the uncropped combined input image
//  - image_B: blue input channel
//  - image_G: green input channel
//  - image_R: red input channel
//  - offset_B: the best offset for the blue channel
//  - offset_G: the best offset for the green channel
//  - out_cropped: output matrix for the cropped image
// return: void
//===============================================================================
void algorithms::crop_image(const cv::Mat &input, const cv::Mat &image_B,
                            const cv::Mat &image_G, const cv::Mat &image_R,
                            const cv::Point2i &offset_B,
                            const cv::Point2i &offset_G, cv::Mat &out_cropped)
{
    // x and y must start at a point that exists in B and G,
    // e.g. offset_B = (1,-4), offset_G = (0,-8) => x=1, y=-4
    int x = cv::max(offset_B.x, offset_G.x);
    int y = cv::max(offset_B.y, offset_G.y);

    // Min width is the x area that overlaps -> reduced by the individual offset
    int width = cv::min(image_B.cols - abs(offset_B.x), image_G.cols - abs(offset_G.x));
    // the width of R could be smaller than of B and G -> reduced by the common offset of B and G
    width = cv::min(width, image_R.cols - abs(x));

    int height = cv::min(image_B.rows - abs(offset_B.y), image_G.rows - abs(offset_G.y));
    height = cv::min(height, image_R.rows - abs(y));

    // The common offset of x and y in B and G could still be negative, e.g. x=-1, y=-4
    // but the red channel always starts at x=0, y=0 -> ensure we won't access non-existing pixels in R
    x = cv::max(0, x);
    y = cv::max(0, y);

    input(cv::Rect(x, y, width, height)).copyTo(out_cropped);
}

//===============================================================================
// bonus()
//-------------------------------------------------------------------------------
// TODO: 1) Blur the image.
//       2) Apply a cartoon filter by discretize the HSV color space using the LUT.
//       3) Add the edges from the original image. NOTE: For the edge
//       detection, you should use the functions from the main task.
//
// parameters:
//  - image: the image to be cartoonized
//  - edge_threshold: intensities above this value are edges
//  - lut: lookup table for discretization
//  - output: output matrix for the final image
// return: void
//===============================================================================
void algorithms::bonus(const cv::Mat &image, const int &edge_threshold,
                       const int (&lut)[256], cv::Mat &output)
{
    // Apply bilateral filter to blur the image
    cv::Mat output_filter = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::bilateralFilter(image, output_filter, 7, 50, 50);

    // Transform the blurred image from RGB to HSV
    cv::Mat hsv_image;
    output_filter.convertTo(hsv_image, CV_8UC3);
    cv::cvtColor(hsv_image, hsv_image, cv::COLOR_BGR2HSV);

    // Get the individual color channels
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv_image, hsv_channels); // 0: H | 1: S | 2: V

    // Replace the color of each pixel in all 3 color channels with the LUT value
    for (int x = 0; x < image.rows; x++)
    {
        for (int y = 0; y < image.cols; y++)
        {
            hsv_channels[0].at<uchar>(x, y) = lut[hsv_channels[0].at<uchar>(x, y)];
            hsv_channels[1].at<uchar>(x, y) = lut[hsv_channels[1].at<uchar>(x, y)];
            hsv_channels[2].at<uchar>(x, y) = lut[hsv_channels[2].at<uchar>(x, y)];
        }
    }

    // Get a grayscale version of the original image and find the edges
    cv::Mat output_gray = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::cvtColor(image, output_gray, cv::COLOR_BGR2GRAY);

    cv::Mat gradient_image = cv::Mat::zeros(output_gray.size(), CV_8UC1);
    cv::Mat binary_image = cv::Mat::zeros(output_gray.size(), CV_8UC1);
    compute_gradient(output_gray, gradient_image);
    compute_binary(gradient_image, edge_threshold, binary_image);

    // Blur the edge image with a Gaussian filter
    cv::Mat gaussian_image = cv::Mat::zeros(output_gray.size(), CV_8UC1);
    cv::GaussianBlur(binary_image, gaussian_image, cv::Size(3, 3), 0.6);

    // Combine each pixel of the blurred edge image with each pixel of the V channel according to the formula:
    // value_new = value_old * (1.0 - edge_val / 255.0)
    for (int x = 0; x < image.rows; x++)
    {
        for (int y = 0; y < image.cols; y++)
        {
            uchar edge_val = gaussian_image.at<uchar>(x, y);
            uchar value_new = hsv_channels[2].at<uchar>(x, y) * (1.0 - edge_val / 255.0);

            hsv_channels[2].at<uchar>(x, y) = value_new;
        }
    }

    // Merge the 3 channels back into one HSV image and convert it back to RGB
    cv::merge(hsv_channels, hsv_image);
    cv::cvtColor(hsv_image, output, cv::COLOR_HSV2BGR);
}
