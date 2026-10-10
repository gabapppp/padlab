#include "dataset.h"
#include <QCoreApplication>
#include <QTemporaryDir>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <stdexcept>
static int checks=0;
static void check(bool ok,const char* s){checks++;if(!ok)throw std::runtime_error(s);}
template<class F>void rejects(F fn,const char* s){bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}check(rejected,s);}
static QJsonObject exactConfig(){auto c=pad::defaultConfig();auto threshold=pad::defaultStep("threshold");auto p=threshold["params"].toObject();p["otsu"]=0;threshold["params"]=p;c["steps"]=QJsonArray{pad::defaultStep("gray"),threshold};return c;}
int main(int argc,char** argv){QCoreApplication app(argc,argv);try{
 auto c=exactConfig();cv::Mat image=cv::Mat::zeros(40,60,CV_8UC3);cv::rectangle(image,cv::Rect(10,10,20,15),cv::Scalar(255,255,255),cv::FILLED);auto result=pad::run(image,c);check(result.boxes.size()==1&&result.boxes[0]==cv::Rect(10,10,20,15),"exact pad bounds");check(result.stages.size()==2,"stage capture");auto perfect=pad::evaluateMask(result.mask,result.mask);check(perfect.iou==1&&perfect.dice==1,"perfect metrics");
 cv::Mat shifted=cv::Mat::zeros(result.mask.size(),CV_8UC1);cv::rectangle(shifted,cv::Rect(11,10,20,15),255,cv::FILLED);auto m=pad::evaluateMask(result.mask,shifted);check(std::abs(m.iou-285.0/315)<1e-10&&m.fp==15&&m.fn==15,"shifted mask IoU");
 auto empty=cv::Mat::zeros(40,60,CV_8UC1);check(pad::evaluateMask(empty,empty).iou==1,"empty masks");check(pad::evaluateMask(empty,result.mask).iou==0,"missed mask");rejects([&]{pad::evaluateMask(result.mask,cv::Mat::zeros(2,2,CV_8UC1));},"size mismatch rejected");
 auto detections=pad::evaluateBoxes({{10,10,20,15},{10,10,20,15}},{{10,10,20,15}},.99);check(detections.tp==1&&detections.fp==1&&detections.fn==0,"duplicate boxes penalized");check(pad::evaluateBoxes({{11,10,20,15}},{{10,10,20,15}},.99).tp==0,"99 percent threshold");
 // A greedy match would lose one TP: p0 overlaps both targets, p1 only target0.
 auto bipartite=pad::evaluateBoxes({{0,0,15,10},{0,0,10,10}},{{0,0,10,10},{5,0,10,10}},.6);check(bipartite.tp==2,"maximum matching");
 auto invalid=c;invalid["schema_version"]=9;rejects([&]{pad::validate(invalid);},"schema rejected");invalid=c;auto blur=pad::defaultStep("gaussian");auto bp=blur["params"].toObject();bp["kernel"]=4;blur["params"]=bp;invalid["steps"]=QJsonArray{pad::defaultStep("gray"),blur,pad::defaultStep("threshold")};rejects([&]{pad::validate(invalid);},"even kernel rejected");
 invalid=c;invalid["steps"]=QJsonArray{pad::defaultStep("threshold")};rejects([&]{pad::validate(invalid);},"invalid step order");invalid=c;invalid["steps"]=QJsonArray{pad::defaultStep("gray"),pad::defaultStep("threshold"),pad::defaultStep("gaussian")};rejects([&]{pad::validate(invalid);},"nonbinary final rejected");
 cv::Mat ring=image.clone();cv::rectangle(ring,cv::Rect(15,14,4,4),cv::Scalar(0,0,0),cv::FILLED);auto fill=c;auto steps=c["steps"].toArray();steps.append(pad::defaultStep("fill_holes"));fill["steps"]=steps;check(cv::countNonZero(pad::run(ring,fill).mask)==300,"hole fill");
 QTemporaryDir temp;check(temp.isValid(),"temporary directory");QDir d(temp.path());d.mkdir("images");d.mkdir("gt");d.mkdir("boxes");auto file=d.filePath("images/pad.png");pad::writeImage(file,image);pad::writeImage(d.filePath("gt/pad.png.png"),result.mask);pad::saveJson(d.filePath("boxes/pad.png.json"),pad::boxesJson(result.boxes));check(pad::readBoxes(d.filePath("boxes/pad.png.json"),image.size()).size()==1,"box roundtrip");
 pad::saveJson(d.filePath("config.json"),c);check(pad::loadConfig(d.filePath("config.json"))==c,"config roundtrip");
 std::atomic_bool cancel{false};auto batch=pad::runFolder(d.filePath("images"),d.filePath("gt"),d.filePath("boxes"),d.filePath("run"),c,cancel);check(batch.rows.size()==1&&batch.rows[0].passed,"batch mask and boxes");check(batch.summary["mask_micro_iou"].toDouble()==1,"aggregate metrics");check(QFile::exists(d.filePath("run/report.csv")),"csv exists");
 rejects([&]{pad::runFolder(d.filePath("images"),{}, {},d.filePath("run"),c,cancel);},"existing output rejected");rejects([&]{pad::runFolder(d.filePath("images"),{}, {},d.filePath("images/results"),c,cancel);},"dataset writes rejected");
 auto unlabeled=pad::runFolder(d.filePath("images"),{}, {},d.filePath("unlabeled"),c,cancel);check(unlabeled.summary["mask_micro_iou"].isNull()&&unlabeled.rows[0].status=="unlabeled","missing GT not scored");
 pad::writeImage(d.filePath("gt/pad.png.png"),cv::Mat::zeros(2,2,CV_8UC1));auto badgt=pad::runFolder(d.filePath("images"),d.filePath("gt"),{},d.filePath("badgt"),c,cancel);check(badgt.rows[0].status=="error"&&badgt.summary["mask_labeled"].toInt()==0,"invalid GT excluded");
 QFile corrupt(d.filePath("images/corrupt.jpg"));corrupt.open(QIODevice::WriteOnly);corrupt.write("not an image");corrupt.close();auto corruptBatch=pad::runFolder(d.filePath("images"),{}, {},d.filePath("corrupt"),c,cancel);check(corruptBatch.summary["errors"].toInt()==1&&corruptBatch.rows.size()==2,"corrupt image isolated");
 cancel.store(true);auto cancelled=pad::runFolder(d.filePath("images"),{}, {},d.filePath("cancel"),c,cancel);check(cancelled.cancelled&&cancelled.rows.empty()&&QFile::exists(d.filePath("cancel/summary.json")),"cancelled report");
 std::cout<<checks<<" checks passed\n";return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
