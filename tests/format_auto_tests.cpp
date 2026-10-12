#include "dataset.h"
#include <QCoreApplication>
#include <QTemporaryDir>
#include <QFile>
#include <QDir>
#include <QJsonArray>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <limits>
#include <stdexcept>
static int checks=0;
static void check(bool ok,const char* label){++checks;if(!ok)throw std::runtime_error(label);}
template<class F>static void rejects(F fn,const char* label){bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}check(rejected,label);}
static void bytesFile(const QString& path,const QByteArray& bytes){QFile f(path);check(f.open(QIODevice::WriteOnly)&&f.write(bytes)==bytes.size(),"write fixture");}
static void encoded(const QString& path,const std::string& extension,const cv::Mat& image){std::vector<uchar> bytes;check(cv::imencode(extension,image,bytes),"encode fixture");bytesFile(path,QByteArray(reinterpret_cast<const char*>(bytes.data()),int(bytes.size())));}
static QJsonObject step(const QString& type,const QJsonObject& overrides={}){auto s=pad::defaultStep(type);auto p=s["params"].toObject();for(auto i=overrides.begin();i!=overrides.end();++i)p[i.key()]=i.value();s["params"]=p;return s;}
static QJsonObject config(const QJsonArray& steps){auto c=pad::defaultConfig();c["steps"]=steps;return c;}
int main(int argc,char** argv){QCoreApplication app(argc,argv);try{
 QTemporaryDir temp;QDir dir(temp.path());cv::Mat color(256,256,CV_8UC3,cv::Scalar(25,80,130));
 cv::rectangle(color,{60,60,100,90},{180,200,220},cv::FILLED);
 for(const auto& ext:{"bmp","png","tiff","webp","jp2","ppm","ras"}){
  QString path=dir.filePath("ảnh."+QString(ext).toUpper());encoded(path,"."+std::string(ext),color);
  auto loaded=pad::loadImage(path);check(loaded.image.type()==CV_8UC3&&loaded.image.size()==color.size(),"format decode dimensions/type");
  if(QString(ext)!="jp2")check(cv::norm(loaded.image,color,cv::NORM_INF)==0,"lossless pixel roundtrip");
  check(pad::imageFiles(temp.path()).contains(path),"uppercase batch discovery");
 }
 cv::Mat gray;cv::cvtColor(color,gray,cv::COLOR_BGR2GRAY);
 for(const auto& ext:{"pgm","pbm"}){QString path=dir.filePath("gray."+QString(ext));encoded(path,"."+std::string(ext),gray);check(pad::loadImage(path).image.size()==gray.size(),"portable gray formats");}
 cv::Mat highDepth(2,3,CV_16U);highDepth.at<ushort>(0,0)=0;highDepth.at<ushort>(0,1)=257;highDepth.at<ushort>(0,2)=65535;highDepth.row(1).setTo(32768);
 encoded(dir.filePath("16bit.tiff"),".tiff",highDepth);auto high=pad::loadImage(dir.filePath("16bit.tiff"));
 check(high.image.at<cv::Vec3b>(0,1)[0]==1&&high.image.at<cv::Vec3b>(0,2)[0]==255&&high.image.at<cv::Vec3b>(1,0)[0]==128,"16-bit fixed mapping");
 check(high.metadata["decoded_depth"]=="16-bit unsigned","depth metadata");
 rejects([&]{pad::readImage(dir.filePath("16bit.tiff"),true);},"no GT normalization");
 cv::Mat floats(3,4,CV_32F,cv::Scalar(-2));floats.at<float>(0,0)=2;encoded(dir.filePath("float.pfm"),".pfm",floats);
 auto floating=pad::loadImage(dir.filePath("float.pfm"));check(floating.image.at<cv::Vec3b>(0,0)[0]==255&&floating.image.at<cv::Vec3b>(2,0)[0]==0,"float min/max");
 floats.setTo(4);encoded(dir.filePath("constant.pfm"),".pfm",floats);check(cv::countNonZero(pad::readImage(dir.filePath("constant.pfm")).reshape(1))==0,"constant float policy");
 floats.at<float>(0,0)=std::numeric_limits<float>::quiet_NaN();encoded(dir.filePath("nan.pfm"),".pfm",floats);rejects([&]{pad::loadImage(dir.filePath("nan.pfm"));},"nonfinite image rejected");
 cv::Mat hdr; color.convertTo(hdr,CV_32F,1.0/255);encoded(dir.filePath("light.hdr"),".hdr",hdr);check(pad::loadImage(dir.filePath("light.hdr")).image.size()==color.size(),"HDR decode");
 cv::Mat alpha(2,2,CV_8UC4,cv::Scalar(100,200,240,0));alpha.at<cv::Vec4b>(0,1)[3]=255;encoded(dir.filePath("alpha.png"),".png",alpha);
 auto a=pad::readImage(dir.filePath("alpha.png"));check(a.at<cv::Vec3b>(0,0)==cv::Vec3b(0,0,0)&&a.at<cv::Vec3b>(0,1)==cv::Vec3b(100,200,240),"alpha composite on black");
 // Real uncommon-codec fixtures: uncompressed TGA, planar RLE PCX and GIF.
 QByteArray tga(18,0);tga[2]=2;tga[12]=2;tga[14]=1;tga[16]=24;tga[17]=32;tga.append(QByteArray::fromHex("0000ff00ff00"));bytesFile(dir.filePath("rare.tga"),tga);
 auto t=pad::loadImage(dir.filePath("rare.tga"));check(t.image.at<cv::Vec3b>(0,0)==cv::Vec3b(0,0,255)&&t.image.at<cv::Vec3b>(0,1)==cv::Vec3b(0,255,0),"TGA codec pixels");
 QByteArray pcx(128,0);pcx[0]=10;pcx[1]=5;pcx[2]=1;pcx[3]=8;pcx[8]=1;pcx[65]=3;pcx[66]=2;pcx[68]=1;pcx.append(QByteArray::fromHex("640000640000"));bytesFile(dir.filePath("rare.pcx"),pcx);
 auto pc=pad::loadImage(dir.filePath("rare.pcx"));check(pc.image.at<cv::Vec3b>(0,0)==cv::Vec3b(0,0,100)&&pc.image.at<cv::Vec3b>(0,1)==cv::Vec3b(0,100,0),"PCX codec pixels");
 bytesFile(dir.filePath("tiny.gif"),QByteArray::fromBase64("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"));check(pad::loadImage(dir.filePath("tiny.gif")).image.size()==cv::Size(1,1),"GIF decode");
 bytesFile(dir.filePath("broken.bmp"),"invalid");rejects([&]{pad::loadImage(dir.filePath("broken.bmp"));},"corrupt codec reports error");
 encoded(dir.filePath("unknown.data"),".png",color);check(pad::loadImage(dir.filePath("unknown.data")).image.size()==color.size(),"content sniff manual open");
 std::vector<cv::Mat> pages{gray,cv::Mat::zeros(gray.size(),gray.type())};check(cv::imwritemulti(dir.filePath("pages.tiff").toStdString(),pages),"write multipage TIFF");
 auto page=pad::loadImage(dir.filePath("pages.tiff"));check(page.metadata["frames"]=="first page/frame only"&&page.image.at<cv::Vec3b>(0,0)[0]==gray.at<uchar>(0,0),"first-page policy");
 cv::Mat bright(100,120,CV_8UC3,cv::Scalar(20,20,20));cv::rectangle(bright,{30,25,60,50},{190,190,190},cv::FILLED);
 auto c=config({step("gray"),step("auto_threshold")});auto result=pad::run(bright,c);check(result.boxes==std::vector<cv::Rect>{{30,25,60,50}},"auto bright foreground exact");
 cv::Mat dark;cv::bitwise_not(bright,dark);auto inverse=pad::run(dark,c);check(cv::norm(result.mask,inverse.mask,cv::NORM_INF)==0,"auto dark polarity exact");
 check(inverse.automatic[0].toObject()["values"].toObject()["inverse"].toBool(),"auto threshold decisions recorded");
 auto fixed=config({step("gray"),step("auto_threshold",{{"polarity",1}})});check(pad::run(dark,fixed).mask.at<uchar>(0,0)==255,"manual polarity override");
 auto triangle=config({step("gray"),step("auto_threshold",{{"method",1}})});check(!pad::run(bright,triangle).mask.empty(),"Triangle mode");
 for(const auto& type:{"auto_contrast","auto_gamma"}){
  auto chain=config({step("gray"),step(type),step("auto_threshold")});auto r=pad::run(bright,chain);
  check(r.stages[1].image.at<uchar>(30,40)>190,"automatic tone adjustment");check(r.automatic.size()==2,"tone and threshold telemetry");
  check(pad::run(bright,chain,false).automatic==r.automatic,"telemetry without captured images");
 }
 for(int value:{0,128,255})for(int preset=0;preset<3;preset++){
  cv::Mat flat(64,64,CV_8UC3,cv::Scalar::all(value));auto r=pad::run(flat,pad::automaticConfig(preset));check(cv::countNonZero(r.mask)==0,"auto presets flat input empty");
 }
 auto edges=config({step("gray"),step("auto_canny")});check(cv::countNonZero(pad::run(bright,edges).mask)>0,"auto Canny sees edges");
 rejects([&]{pad::validate(config({step("auto_gamma"),step("threshold")}));},"auto tone rejects color");
 rejects([&]{pad::validate(config({step("gray"),step("threshold"),step("auto_contrast")}));},"auto tone rejects binary");
 dir.mkdir("batch");encoded(dir.filePath("batch/one.BMP"),".bmp",bright);encoded(dir.filePath("batch/two.pgm"),".pgm",gray);
 std::atomic_bool cancel{false};auto batch=pad::runFolder(dir.filePath("batch"),{},{},dir.filePath("output"),c,cancel);
 check(batch.rows.size()==2&&batch.summary["errors"].toInt()==0,"mixed format batch");check(!batch.rows[0].inputMetadata.isEmpty()&&!pad::rowJson(batch.rows[0])["automatic"].toArray().isEmpty(),"batch decoder and auto report");
 std::cout<<checks<<" format / automatic checks passed\n";return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
