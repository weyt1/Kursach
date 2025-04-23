//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>

int main() {
    const std::string input_path = "markers.jpg";
    cv::Mat inputImage = cv::imread(input_path);
    
    if (inputImage.empty()) {
        std::cerr << "Error: Could not load image at " << input_path << "!" << std::endl;
        return -1;
    }

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

    if (!markerIds.empty()) {
        cv::Mat outputImage = inputImage.clone();
        cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);
        
        std::cout << "Detected markers: ";
        for (int id : markerIds) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        const std::string output_path = "Detected Markers.png";
        cv::imshow(output_path, outputImage);
        cv::imwrite(output_path, outputImage);
        cv::waitKey(0);
    } else {
        std::cout << "No markers found!" << std::endl;
    }

    return 0;
}