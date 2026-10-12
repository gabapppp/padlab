#pragma once
#include <QJsonObject>
#include <QStringList>
#include <opencv2/core.hpp>
namespace pad {
struct LoadedImage { cv::Mat image; QJsonObject metadata; QString description; };
QStringList imageExtensions();
QString imageFileFilter();
LoadedImage loadImage(const QString& path);
}
