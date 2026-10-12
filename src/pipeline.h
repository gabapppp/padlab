#pragma once
#include <QJsonObject>
#include <QJsonArray>
#include "imageio.h"
#include "automatic.h"
#include <QString>
#include <QStringList>
#include <opencv2/core.hpp>
#include <vector>
#include <functional>
namespace pad {
struct Parameter { QString key; double value, min, max; bool integer; };
struct Definition { QString type, label; std::vector<Parameter> parameters; };
const std::vector<Definition>& definitions();
QJsonObject defaultStep(const QString& type);
QJsonObject defaultConfig();
QJsonObject precisionConfig(bool lowContrast=false);
void validate(const QJsonObject& config);
QJsonObject loadConfig(const QString& path);
void saveJson(const QString& path, const QJsonObject& value);
cv::Mat readImage(const QString& path, bool mask=false);
void writeImage(const QString& path, const cv::Mat& image);
struct EdgePoint {
 cv::Point2f seed, position, normal;
 float strength=0, before=0, peak=0, after=0;
 bool localized=false, refined=false;
};
struct EdgeContour { std::vector<EdgePoint> points; int parent=-1; bool hole=false; };
struct Stage {
 QString name; QJsonObject automatic; cv::Mat image; double ms=0; int stepIndex=-1;
 std::vector<EdgeContour> boundaries;
};
struct Result { QJsonArray automatic; cv::Mat mask, overlay; std::vector<cv::Rect> boxes; std::vector<Stage> stages; std::vector<EdgeContour> boundaries; bool hasBoundaries=false, subpixel=false; double ms=0; };
Result run(const cv::Mat& image, const QJsonObject& config, bool stages=true,
           const std::function<bool()>& cancelled={});
struct Metrics { double iou=0, dice=0, precision=0, recall=0, accuracy=0; double tp=0,fp=0,fn=0,tn=0; };
Metrics evaluateMask(const cv::Mat& prediction, const cv::Mat& truth);
Metrics fromCounts(double tp,double fp,double fn,double tn);
struct DetectionMetrics { int tp=0,fp=0,fn=0; double meanMatchedIou=0; };
DetectionMetrics evaluateBoxes(const std::vector<cv::Rect>& predicted,
 const std::vector<cv::Rect>& truth, double threshold);
std::vector<cv::Rect> maskBoxes(const cv::Mat& mask);
QJsonObject boundariesJson(const Result& result);
cv::Mat boundaryOverlay(const cv::Mat& image,const std::vector<EdgeContour>& contours);
QJsonObject boxesJson(const std::vector<cv::Rect>& boxes);
std::vector<cv::Rect> readBoxes(const QString& path, cv::Size size);
}
