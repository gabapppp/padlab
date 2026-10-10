#pragma once
#include "pipeline.h"
#include <atomic>
namespace pad {
struct Row { QString file,status,error,maskSource,boxSource; int objects=0;double ms=0;bool hasMask=false,hasBoxes=false,passed=false;Metrics mask;DetectionMetrics detection; };
struct Batch { std::vector<Row> rows; QJsonObject summary; bool cancelled=false; };
QStringList imageFiles(const QString& folder);
Row assess(const QString& file,const Result& result,const QJsonObject& config,const QString& maskPath,const QString& boxPath);
Batch runFolder(const QString& input,const QString& masks,const QString& boxes,const QString& output,const QJsonObject& config,
 std::atomic_bool& cancel,const std::function<void(int,int)>& progress={});
QJsonObject rowJson(const Row& row);
void exportStageImages(const QString& directory,const cv::Mat& original,const Result& result,const QJsonObject& config);
QString metricsText(const Row& row);
}
