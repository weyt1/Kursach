#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>

void readCameraParamsFromCommandLine(const cv::CommandLineParser& parser, cv::Mat& camMatrix, cv::Mat& distCoeffs) {
    camMatrix = (cv::Mat_<double>(3,3) << 1.35076413e+03, 0, 4.13808477e+02, 0, 1.71921253e+03, 1.98116410e+02, 0, 0, 1);
    distCoeffs = (cv::Mat_<double>(5,1) << 
         0.2470824, 
         -0.27398596, 
         -0.02732488, 
         -0.02180864, 
         -0.30900811);

}

cv::Ptr<cv::aruco::Dictionary> readDictionaryFromCommandLine(const cv::CommandLineParser& parser) {
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
}

cv::Ptr<cv::aruco::DetectorParameters> readDetectorParamsFromCommandLine(const cv::CommandLineParser& parser) {
    return cv::aruco::DetectorParameters::create();
}

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv,
        "{w |5|Number of markers in X direction}"
        "{h |7|Number of markers in Y direction}"
        "{l |0.03|Marker length (in meters)}"
        "{s |0.01|Marker separation (in meters)}"
        "{r||Show rejected markers}"
        "{rs||Use refined strategy}"
        "{ci|http://10.160.98.77:8080/video|Camera URL}"
        "{v||Input video file}"
        "{help||Help}");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    int markersX = parser.get<int>("w");
    int markersY = parser.get<int>("h");
    float markerLength = parser.get<float>("l");
    float markerSeparation = parser.get<float>("s");
    bool showRejected = parser.has("r");
    bool refindStrategy = parser.has("rs");
    std::string cameraSource = parser.get<std::string>("ci");

    cv::Mat camMatrix, distCoeffs;
    readCameraParamsFromCommandLine(parser, camMatrix, distCoeffs);
    auto dictionary = readDictionaryFromCommandLine(parser);
    auto detectorParams = readDetectorParamsFromCommandLine(parser);

    cv::String video;
    if (parser.has("v")) {
        video = parser.get<cv::String>("v");
    }

    if (!parser.check()) {
        parser.printErrors();
        return 1;
    }

    cv::VideoCapture inputVideo;
    int waitTime;
    
    if (!video.empty()) {
        inputVideo.open(video);
        waitTime = 0;
    } else {
        inputVideo.open(cameraSource);
        
        if (!inputVideo.isOpened()) {
            std::cerr << "Failed to open phone camera" << std::endl;
        }
        waitTime = 10;
    }
    
    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(
        markersX, markersY, markerLength, markerSeparation, dictionary);
        
    double totalTime = 0;
    int totalIterations = 0;
    
    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        
        double tick = (double)cv::getTickCount();
        
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners, rejected;
        cv::Vec3d rvec, tvec;
        
        cv::aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
        
        if (refindStrategy) {
            cv::aruco::refineDetectedMarkers(image, board, corners, ids, rejected, camMatrix, distCoeffs);
        }

        int markersOfBoardDetected = 0;
        if (!ids.empty()) {
            markersOfBoardDetected = cv::aruco::estimatePoseBoard(
                corners, ids, board, camMatrix, distCoeffs, rvec, tvec);
        }

        double currentTime = ((double)cv::getTickCount() - tick) / cv::getTickFrequency();
        totalTime += currentTime;
        totalIterations++;

        if (totalIterations % 30 == 0) {
            std::cout << "Detection Time = " << currentTime * 1000 << " ms "
                     << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << std::endl;
        }

        image.copyTo(imageCopy);
        if (!ids.empty()) {
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
        }

        if (showRejected && !rejected.empty()) {
            cv::aruco::drawDetectedMarkers(imageCopy, rejected, cv::noArray(), cv::Scalar(100, 0, 255));
        }

        if (markersOfBoardDetected > 0) {
            cv::drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvec, tvec, 3);
        }

        cv::imshow("out", imageCopy);
        char key = (char)cv::waitKey(waitTime);
        if (key == 27) break;
    }

    return 0;
}