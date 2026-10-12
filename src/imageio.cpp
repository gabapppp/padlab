#include "imageio.h"
#include <QFile>
#include <QImageReader>
#include <QBuffer>
#include <QFileInfo>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <cmath>

namespace pad {
QStringList imageExtensions() {
 QStringList extensions{"bmp","dib","jpeg","jpg","jpe","jp2","png","webp",
  "tif","tiff","pbm","pgm","ppm","pnm","pfm","sr","ras","hdr","pic","exr"};
 for(const auto& format:QImageReader::supportedImageFormats())extensions<<QString::fromLatin1(format).toLower();
 extensions.removeDuplicates();extensions.sort();return extensions;
}
QString imageFileFilter() {
 QStringList patterns;for(const auto& extension:imageExtensions())patterns<<"*."+extension;
 return "Images ("+patterns.join(' ')+");;All files (*)";
}
LoadedImage loadImage(const QString& path) {
 QFile file(path);
 if(!file.open(QIODevice::ReadOnly))throw std::runtime_error(("Cannot read "+path+": "+file.errorString()).toStdString());
 QByteArray bytes=file.readAll();
 if(bytes.isEmpty())throw std::runtime_error("Image file is empty");
 cv::Mat source;QString decoder="OpenCV",decodeError;
 try {source=cv::imdecode(std::vector<uchar>(bytes.begin(),bytes.end()),cv::IMREAD_UNCHANGED);}
 catch(const cv::Exception& e){decodeError=QString::fromUtf8(e.what());}
 if(source.empty()) {
  QBuffer buffer(&bytes);buffer.open(QIODevice::ReadOnly);QImageReader reader(&buffer);
  reader.setDecideFormatFromContent(true);reader.setAutoTransform(false);
  auto image=reader.read();QString qtFormat=QString::fromLatin1(reader.format());QString qtError=reader.errorString();
  if(image.isNull()) {
   buffer.seek(0);QImageReader hinted(&buffer,QFileInfo(path).suffix().toLower().toLatin1());hinted.setAutoTransform(false);
   image=hinted.read();qtFormat=QString::fromLatin1(hinted.format());qtError=hinted.errorString();
  }
  if(image.isNull())throw std::runtime_error(("Cannot decode image: "+path+"\nQt: "+qtError+
    (decodeError.isEmpty()?QString():"\nOpenCV: "+decodeError)+"\nA filename extension alone does not guarantee codec support. Camera RAW and DICOM require prior conversion.").toStdString());
  decoder="Qt / "+qtFormat+" (RGBA8 decode)";
  image=image.convertToFormat(QImage::Format_RGBA8888);
  cv::Mat rgba(image.height(),image.width(),CV_8UC4,image.bits(),image.bytesPerLine());
  cv::cvtColor(rgba,source,cv::COLOR_RGBA2BGRA);
 }
 const int depth=source.depth(),channels=source.channels();
 if(channels!=1&&channels!=3&&channels!=4)throw std::runtime_error("Only grayscale, RGB and RGBA images are supported");
 QString depthName;switch(depth){case CV_8U:depthName="8-bit unsigned";break;case CV_16U:depthName="16-bit unsigned";break;
 case CV_16S:depthName="16-bit signed";break;case CV_32F:depthName="32-bit float";break;case CV_64F:depthName="64-bit float";break;
 default:throw std::runtime_error("Unsupported image pixel depth");}
 double low=0,high=255,scale=1,offset=0;QString mapping="unchanged";
 if(depth==CV_16U){high=65535;scale=255.0/65535;mapping="fixed unsigned 16-bit range to 0..255";}
 else if(depth!=CV_8U){
  if(!cv::checkRange(source,true,nullptr,-1e100,1e100))throw std::runtime_error("Image contains non-finite or unsupported extreme intensities");
  cv::minMaxLoc(source.reshape(1),&low,&high);
  scale=high>low?255/(high-low):0;offset=-low*scale;
  mapping="per-image min/max to 0..255; constant image maps to zero";
 }
 cv::Mat normalized;source.convertTo(normalized,CV_8U,scale,offset);
 if(channels==4&&(depth==CV_32F||depth==CV_64F)){
  cv::Mat alpha;cv::extractChannel(source,alpha,3);alpha.convertTo(alpha,CV_8U,255);cv::insertChannel(alpha,normalized,3);
 }
 LoadedImage result;
 if(channels==1)cv::cvtColor(normalized,result.image,cv::COLOR_GRAY2BGR);
 else if(channels==3)result.image=normalized;
 else {
  result.image.create(source.size(),CV_8UC3);
  for(int y=0;y<source.rows;y++)for(int x=0;x<source.cols;x++){
   const auto pixel=normalized.at<cv::Vec4b>(y,x);auto& target=result.image.at<cv::Vec3b>(y,x);
   for(int c=0;c<3;c++)target[c]=uchar((int(pixel[c])*pixel[3]+127)/255);
  }
 }
 result.metadata={{"decoder",decoder},{"decoded_depth",depthName},{"decoded_channels",channels},
  {"width",source.cols},{"height",source.rows},{"working_type","8-bit BGR"},{"mapping",mapping},
  {"mapping_low",low},{"mapping_high",high},{"scale",scale},{"offset",offset},
  {"orientation","stored pixel order; EXIF rotation disabled"},{"frames","first page/frame only"},
  {"alpha","composited on black"}};
 result.description=QString("%1 × %2 · %3 · %4 · 8-bit workspace").arg(source.cols).arg(source.rows).arg(depthName,decoder);
 if(depth!=CV_8U)result.description+=" · "+mapping;
 return result;
}
}
