#pragma once
#include <QJsonObject>
#include <opencv2/core.hpp>
namespace pad {
cv::Mat automaticStep(const QString& type,const cv::Mat& image,const QJsonObject& parameters,QJsonObject& chosen);
QJsonObject automaticConfig(int preset=0);
}
