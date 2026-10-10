#include "advanced.h"
#include "dataset.h"
#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <QFile>
#include <QDir>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <stdexcept>
namespace {
int checks=0;
void require(bool value,const char* message){checks++;if(!value)throw std::runtime_error(message);}
template<class F>void rejects(F f,const char* message){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,message);}
QJsonObject config(std::initializer_list<const char*> types){auto c=pad::defaultConfig();QJsonArray steps;for(auto t:types)steps.append(pad::defaultStep(t));c["steps"]=steps;return c;}
}
int main(int argc,char** argv){QCoreApplication app(argc,argv);try{
 for(bool horizontal:{false,true})for(double phase:{0.2,0.35,0.7}){
  const double edge=42+phase;cv::Mat gray(96,128,CV_8U);
  for(int y=0;y<gray.rows;y++)for(int x=0;x<gray.cols;x++)gray.at<uchar>(y,x)=cv::saturate_cast<uchar>(25+210*0.5*(1+std::erf(((horizontal?y:x)-edge)/(std::sqrt(2.)*1.2))));
  cv::Mat source,mask;cv::cvtColor(gray,source,cv::COLOR_GRAY2BGR);cv::threshold(gray,mask,128,255,cv::THRESH_BINARY);
  auto coarse=pad::localizeEdges(mask,source,3,0.6,3,3);auto fine=coarse;pad::refineSubpixel(fine,0.001);
  double before=0,after=0;int count=0;
  for(size_t c=0;c<fine.size();c++)for(size_t i=0;i<fine[c].points.size();i++){
   const auto& a=coarse[c].points[i];const auto& b=fine[c].points[i];const float cross=horizontal?b.seed.x:b.seed.y;
   if(b.refined&&std::abs((horizontal?b.seed.y:b.seed.x)-edge)<3&&cross>12&&cross<75){before+=std::abs((horizontal?a.position.y:a.position.x)-edge);after+=std::abs((horizontal?b.position.y:b.position.x)-edge);count++;}
   require(std::isfinite(b.position.x)&&std::isfinite(b.position.y),"finite boundary coordinate");
  }
  require(count>40,"enough refined interior points");before/=count;after/=count;
  std::cout<<"edge="<<edge<<" horizontal="<<horizontal<<" coarse error="<<before<<" refined error="<<after<<" px\n";
  require(after<before&&after<0.10,"subpixel improves known fractional straight edge");
 }
 cv::Mat original(100,130,CV_8UC3,cv::Scalar(90,90,90)),mask=cv::Mat::zeros(original.size(),CV_8U);cv::rectangle(mask,cv::Rect(0,10,80,70),255,-1);cv::circle(mask,cv::Point(40,40),10,0,-1);
 auto weak=pad::localizeEdges(mask,original,3,1,3,3);pad::refineSubpixel(weak,0.001);bool hole=false;
 for(const auto& c:weak){hole|=c.hole;for(const auto& p:c.points)require(!p.localized&&!p.refined&&p.position==p.seed,"weak edge kept at seed");}require(hole,"contour holes retained");
 require(pad::localizeEdges(cv::Mat::zeros(mask.size(),CV_8U),original,3,1,3,3).empty(),"empty mask geometry");
 rejects([&]{pad::localizeEdges(mask,original,3,1,3,3,[]{return true;});},"edge cancellation");
 pad::validate(pad::precisionConfig());pad::validate(pad::precisionConfig(true));
 auto bad=config({"gray","threshold","subpixel"});rejects([&]{pad::validate(bad);},"subpixel without localization rejected");
 bad=config({"gray","threshold","edge_localize","morphology"});rejects([&]{pad::validate(bad);},"stale boundary invalidated by later mask step");
 bad=config({"gray","threshold","edge_localize","subpixel","subpixel"});rejects([&]{pad::validate(bad);},"repeated refinement rejected");
 bad=config({"gray","threshold","guided"});rejects([&]{pad::validate(bad);},"guided binary input rejected");
 cv::Mat flat(80,80,CV_8U,cv::Scalar(90));require(cv::norm(pad::guidedSmooth(flat,4,.01),flat,cv::NORM_INF)==0,"guided preserves constant");require(cv::norm(pad::correctIllumination(flat,20,true),flat,cv::NORM_INF)<=1,"division preserves constant");
 auto light=pad::sauvolaMask(flat,31,.2,128,false),dark=pad::sauvolaMask(flat,31,.2,128,true);require(cv::countNonZero(light)==flat.rows*flat.cols&&cv::countNonZero(dark)==0,"Sauvola polarity");
 cv::Mat noisy(100,160,CV_8U);cv::RNG rng(42);rng.fill(noisy,cv::RNG::NORMAL,60,12);noisy(cv::Rect(80,0,80,100))+=130;
 auto smooth=pad::guidedSmooth(noisy,4,.01);cv::Scalar meanA,stdA,meanB,stdB;cv::meanStdDev(noisy(cv::Rect(10,10,55,80)),meanA,stdA);cv::meanStdDev(smooth(cv::Rect(10,10,55,80)),meanB,stdB);
 require(stdB[0]<stdA[0]*.6,"guided reduces noise");require(cv::mean(smooth(cv::Rect(95,10,55,80)))[0]-meanB[0]>110,"guided retains contrast");
 cv::Mat touching=cv::Mat::zeros(120,180,CV_8U);cv::circle(touching,cv::Point(65,60),35,255,-1);cv::circle(touching,cv::Point(115,60),35,255,-1);cv::Mat color;cv::cvtColor(touching,color,cv::COLOR_GRAY2BGR);auto split=pad::watershedSplit(touching,color,9,.4);require(pad::maskBoxes(split).size()==2,"watershed separates touching disks");cv::Mat outside;cv::bitwise_and(split,~touching,outside);require(cv::countNonZero(outside)==0,"watershed stays inside mask");
 auto c=pad::precisionConfig();auto result=pad::run(color,c);require(result.hasBoundaries&&result.subpixel,"pipeline geometry enabled");auto base=pad::run(color,pad::defaultConfig());require(cv::norm(base.mask,result.mask,cv::NORM_INF)==0,"measurement does not change mask score");
 auto disabled=pad::defaultStep("median");disabled["enabled"]=false;auto stages=c["steps"].toArray();stages.insert(1,disabled);c["steps"]=stages;result=pad::run(color,c);require(result.stages[1].stepIndex==2,"stage indexes include disabled slots");
 auto json=pad::boundariesJson(result);require(json["refined"].toInt()>0,"export refined counts");auto parsed=QJsonDocument::fromJson(QJsonDocument(json).toJson()).object();require(parsed==json,"float JSON roundtrip");
 QTemporaryDir tmp;QString path=QDir(tmp.path()).filePath("steps");pad::exportStageImages(path,color,result,c);require(QFile::exists(path+"/01-gray.png")&&QFile::exists(path+"/03-gaussian.png")&&!QFile::exists(path+"/02-median.png"),"stage export matches enabled indexes");
 require(QFile::exists(path+"/boundaries.json")&&QFile::exists(path+"/stages.json"),"stage export manifests");rejects([&]{pad::exportStageImages(path,color,result,c);},"step export never overwrites");
 QDir d(tmp.path());d.mkdir("input");pad::writeImage(d.filePath("input/sample.png"),color);std::atomic_bool cancel{false};auto batch=pad::runFolder(d.filePath("input"),{},{},d.filePath("run"),c,cancel);require(batch.rows.size()==1&&batch.rows[0].status=="unlabeled","advanced batch succeeds");require(QFile::exists(d.filePath("run/boundaries/sample.png.json")),"batch boundary export");
 std::cout<<checks<<" advanced checks passed\n";return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
