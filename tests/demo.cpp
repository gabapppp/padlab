#include "pipeline.h"
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <opencv2/imgproc.hpp>
#include <iostream>
int main(int argc,char** argv){QCoreApplication app(argc,argv);if(argc!=2)return 1;try{QDir d(QString::fromLocal8Bit(argv[1]));if(d.exists())return 0;QDir().mkpath(d.path());d.mkdir("images");d.mkdir("masks");d.mkdir("boxes");auto c=pad::defaultConfig();auto step=pad::defaultStep("threshold");auto params=step["params"].toObject();params["otsu"]=0;params["value"]=100;step["params"]=params;c["steps"]=QJsonArray{pad::defaultStep("gray"),step};pad::saveJson(d.filePath("pipeline.json"),c);
 for(int i=0;i<4;i++){cv::Mat image(360,520,CV_8UC3,cv::Scalar(25,30,35)),mask=cv::Mat::zeros(image.size(),CV_8UC1);std::vector<cv::Rect> boxes;if(i!=3){cv::Rect r(90+i*12,80+i*9,300,150);boxes.push_back(r);cv::rectangle(image,r,cv::Scalar(190,195,200),cv::FILLED);cv::rectangle(mask,r,255,cv::FILLED);if(i==1)for(int k=0;k<20;k++)cv::circle(image,cv::Point(120+k*12,130),2,cv::Scalar(65,65,65),cv::FILLED);if(i==2)cv::rectangle(image,cv::Rect(r.x,r.y,6,r.height),cv::Scalar(50,50,50),cv::FILLED);}auto name=QString("pad-%1.png").arg(i);pad::writeImage(d.filePath("images/"+name),image);pad::writeImage(d.filePath("masks/"+name+".png"),mask);pad::saveJson(d.filePath("boxes/"+name+".json"),pad::boxesJson(boxes));}std::cout<<"Synthetic demo generated (includes holes, edge loss and an empty image)\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
