#pragma once
#include "pipeline.h"
namespace pad {
cv::Mat correctIllumination(const cv::Mat& gray, double sigma, bool divide);
cv::Mat guidedSmooth(const cv::Mat& gray, int radius, double epsilon);
cv::Mat sauvolaMask(const cv::Mat& gray, int block, double k, double range, bool inverse);
cv::Mat watershedSplit(const cv::Mat& mask, const cv::Mat& original, int peakRadius, double peakFraction);
std::vector<EdgeContour> localizeEdges(const cv::Mat& mask, const cv::Mat& original,
 int radius, double sigma, double minGradient, int normalSpan,
 const std::function<bool()>& cancelled={});
void refineSubpixel(std::vector<EdgeContour>& contours, double minCurvature,
 const std::function<bool()>& cancelled={});
}
