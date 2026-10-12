#include "automatic.h"
#include "pipeline.h"
#include <QJsonArray>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <array>

namespace pad {
static double percentile(const cv::Mat& image,double percent) {
 std::array<size_t,256> histogram{};
 for(int y=0;y<image.rows;y++)for(int x=0;x<image.cols;x++)histogram[image.at<uchar>(y,x)]++;
 const size_t rank=size_t((image.total()-1)*percent/100);size_t count=0;
 for(int i=0;i<256;i++){count+=histogram[i];if(count>rank)return i;}
 return 255;
}
static bool borderForeground(const cv::Mat& mask) {
 size_t foreground=0,total=0;
 for(int y=0;y<mask.rows;y++)for(int x=0;x<mask.cols;x++)if(y==0||x==0||y==mask.rows-1||x==mask.cols-1){total++;foreground+=mask.at<uchar>(y,x)!=0;}
 return foreground>total/2;
}
cv::Mat automaticStep(const QString& type,const cv::Mat& image,const QJsonObject& p,QJsonObject& chosen) {
 cv::Mat out;
 if(type=="auto_contrast") {
  double low=percentile(image,p["clip_percent"].toDouble()),high=percentile(image,100-p["clip_percent"].toDouble());
  chosen={{"low",low},{"high",high},{"changed",high>low}};
  if(high>low)image.convertTo(out,CV_8U,255/(high-low),-low*255/(high-low));else out=image.clone();
 } else if(type=="auto_gamma") {
  double mean=cv::mean(image)[0]/255,gamma=1;
  if(mean>0&&mean<1)gamma=std::clamp(std::log(p["target"].toDouble())/std::log(mean),0.2,5.0);
  cv::Mat lut(1,256,CV_8U);for(int i=0;i<256;i++)lut.at<uchar>(i)=cv::saturate_cast<uchar>(255*std::pow(i/255.0,gamma));
  cv::LUT(image,lut,out);chosen={{"gamma",gamma},{"input_mean",mean},{"target",p["target"]}};
 } else if(type=="auto_threshold") {
  const int method=p["method"].toInt(),polarity=p["polarity"].toInt();
  double threshold=cv::threshold(image,out,0,255,cv::THRESH_BINARY|(method?cv::THRESH_TRIANGLE:cv::THRESH_OTSU));
  // A constant field contains no contrast from which to infer an object.
  double low,high;cv::minMaxLoc(image,&low,&high);bool inverse=false;
  if(low==high)out.setTo(0);
  else {inverse=polarity==2||(polarity==0&&borderForeground(out));if(inverse)cv::bitwise_not(out,out);}
  chosen={{"threshold",threshold},{"method",method?"Triangle":"Otsu"},{"inverse",inverse},{"constant_image",low==high}};
 } else if(type=="auto_polarity") {
  bool inverse=borderForeground(image);out=image.clone();if(inverse)cv::bitwise_not(out,out);
  chosen={{"inverse",inverse},{"rule","majority of outer border is background"}};
 } else if(type=="auto_canny") {
  cv::Mat dx,dy,magnitude;cv::Sobel(image,dx,CV_32F,1,0,3);cv::Sobel(image,dy,CV_32F,0,1,3);cv::magnitude(dx,dy,magnitude);
  std::vector<float> nonzero;nonzero.reserve(image.total());
  for(int y=0;y<image.rows;y++)for(int x=0;x<image.cols;x++){float g=magnitude.at<float>(y,x);if(g>0)nonzero.push_back(g);}
  double high=0,low=0;
  if(nonzero.empty())out=cv::Mat::zeros(image.size(),CV_8U);
  else {
   size_t rank=size_t((nonzero.size()-1)*p["percentile"].toDouble()/100);
   std::nth_element(nonzero.begin(),nonzero.begin()+rank,nonzero.end());
   high=std::max(1.0,double(nonzero[rank])*0.9);low=high*p["low_ratio"].toDouble();
   cv::Canny(image,out,low,high,3,true);
  }
  chosen={{"low",low},{"high",high},{"l2_gradient",true},{"rule","90% of selected nonzero Sobel magnitude percentile"}};
 }
 return out;
}
QJsonObject automaticConfig(int preset) {
 auto config=defaultConfig();QJsonArray steps;
 for(const auto& type:{"gray","auto_contrast","gaussian"})steps.append(defaultStep(type));
 if(preset==1){steps.append(defaultStep("illumination"));steps.append(defaultStep("sauvola"));steps.append(defaultStep("auto_polarity"));}
 else if(preset==2){steps.append(defaultStep("auto_canny"));auto close=defaultStep("morphology");auto p=close["params"].toObject();p["operation"]=1;close["params"]=p;steps.append(close);steps.append(defaultStep("fill_holes"));}
 else steps.append(defaultStep("auto_threshold"));
 steps.append(defaultStep("components"));config["steps"]=steps;return config;
}
}
