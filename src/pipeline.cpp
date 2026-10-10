#include "pipeline.h"
#include "advanced.h"
#include <QFile>
#include <QSaveFile>
#include <QJsonDocument>
#include <QJsonArray>
#include <QElapsedTimer>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <cmath>
#include <algorithm>
namespace pad {
static void fail(const QString& s) { throw std::runtime_error(s.toStdString()); }
const std::vector<Definition>& definitions() {
 static const std::vector<Definition> d={
 {"gray","Grayscale",{}},
 {"channel","Color channel (BGR / HSV / Lab)",{{"space",0,0,2,true},{"channel",0,0,2,true}}},
 {"clahe","CLAHE",{{"clip",2,0.1,40,false},{"grid",8,1,64,true}}},
 {"gaussian","Gaussian blur",{{"kernel",5,1,99,true},{"sigma",0,0,50,false}}},
 {"median","Median blur",{{"kernel",5,3,99,true}}},
 {"bilateral","Bilateral filter",{{"diameter",9,1,31,true},{"color",50,1,255,false},{"space",50,1,255,false}}},
 {"threshold","Threshold (fixed / Otsu)",{{"value",128,0,255,true},{"otsu",1,0,1,true},{"inverse",0,0,1,true}}},
 {"adaptive","Adaptive threshold",{{"block",31,3,199,true},{"c",5,-100,100,false},{"gaussian",1,0,1,true},{"inverse",0,0,1,true}}},
 {"hsv","HSV range",{{"h_min",0,0,179,true},{"h_max",179,0,179,true},{"s_min",0,0,255,true},{"s_max",255,0,255,true},{"v_min",100,0,255,true},{"v_max",255,0,255,true}}},
 {"illumination","Illumination correction",{{"sigma",25,1,200,false},{"divide",0,0,1,true}}},
 {"guided","Guided filter (edge preserving)",{{"radius",4,1,50,true},{"epsilon",0.01,0.0001,1,false}}},
 {"sauvola","Sauvola local threshold",{{"block",31,3,199,true},{"k",0.2,-1,1,false},{"range",128,1,255,false},{"inverse",0,0,1,true}}},
 {"watershed","Watershed split touching regions",{{"peak_radius",7,1,99,true},{"peak_fraction",0.4,0.01,1,false}}},
 {"edge_localize","Edge localization (normal profiles)",{{"radius",3,1,15,true},{"sigma",1,0,5,false},{"min_gradient",3,0.1,255,false},{"normal_span",3,1,20,true},{"source",0,0,1,true}}},
 {"subpixel","Subpixel boundary refinement",{{"min_curvature",0.001,0.0001,255,false}}},
 {"canny","Canny edges",{{"low",50,0,255,false},{"high",150,0,255,false}}},
 {"morphology","Morphology",{{"operation",0,0,5,true},{"shape",2,0,2,true},{"kernel",5,1,99,true},{"iterations",1,1,20,true}}},
 {"invert","Invert",{}},
 {"fill_holes","Fill enclosed holes",{}},
 {"components","Filter connected regions",{{"min_area",100,0,100000000,true},{"max_area",100000000,1,100000000,true},{"min_aspect",0.05,0.001,100,false},{"max_aspect",20,0.001,100,false},{"min_extent",0,0,1,false},{"largest_only",0,0,1,true}}}
 }; return d;
}
static const Definition& definition(const QString& type) {
 for(const auto& d:definitions()) if(d.type==type) return d;
 fail("Unknown step: "+type); return definitions()[0];
}
QJsonObject defaultStep(const QString& type) {
 const auto& d=definition(type); QJsonObject p;
 for(const auto& v:d.parameters) p[v.key]=v.value;
 return {{"type",type},{"enabled",true},{"params",p}};
}
QJsonObject defaultConfig() {
 QJsonArray steps; for(const auto& t:{"gray","gaussian","threshold","morphology","components"}) steps.append(defaultStep(t));
 return {{"schema_version",1},{"steps",steps},{"evaluation",QJsonObject{{"box_iou_threshold",0.99},{"mask_iou_target",0.99}}}};
}
QJsonObject precisionConfig(bool lowContrast) {
 auto c=defaultConfig();
 if(lowContrast) {
  QJsonArray steps;
  for(const auto& type:{"gray","illumination","guided","sauvola","morphology","fill_holes","components"})steps.append(defaultStep(type));
  c["steps"]=steps;
 }
 auto steps=c["steps"].toArray();steps.append(defaultStep("edge_localize"));steps.append(defaultStep("subpixel"));c["steps"]=steps;
 return c;
}
void validate(const QJsonObject& c) {
 for(auto it=c.begin();it!=c.end();++it) if(it.key()!="schema_version"&&it.key()!="steps"&&it.key()!="evaluation") fail("Unknown config key: "+it.key());
 if(!c["schema_version"].isDouble()||c["schema_version"].toDouble()!=1) fail("Unsupported schema_version");
 if(!c["steps"].isArray()||c["steps"].toArray().isEmpty()||c["steps"].toArray().size()>100) fail("Pipeline must have 1..100 steps");
 const auto e=c["evaluation"].toObject();
 if(!c["evaluation"].isObject()||e.size()!=2) fail("evaluation requires box_iou_threshold and mask_iou_target");
 for(const auto& k:{"box_iou_threshold","mask_iou_target"}) if(!e[k].isDouble()||!std::isfinite(e[k].toDouble())||e[k].toDouble()<0||e[k].toDouble()>1) fail(QString("Invalid evaluation: ")+k);
 int mode=3, geometry=0; bool enabled=false;
 for(const auto& v:c["steps"].toArray()) {
  if(!v.isObject()) fail("Each step must be an object"); const auto s=v.toObject();
  if(s.size()!=3||!s["type"].isString()||!s["enabled"].isBool()||!s["params"].isObject()) fail("Step requires type, enabled, params only");
  const auto& d=definition(s["type"].toString()); const auto p=s["params"].toObject();
  if(p.size()!=int(d.parameters.size())) fail("Unexpected/missing parameters in "+d.type);
  for(const auto& f:d.parameters) {
   if(!p[f.key].isDouble()) fail("Missing numeric parameter: "+f.key);
   double x=p[f.key].toDouble(); if(!std::isfinite(x)||x<f.min||x>f.max||(f.integer&&std::floor(x)!=x)) fail("Out of range: "+d.type+"."+f.key);
  }
  for(const auto& k:{"kernel","block"}) if(p.contains(k)&&int(p[k].toDouble())%2==0) fail(d.type+"."+k+" must be odd");
  if(d.type=="hsv") { for(const auto& k:{"h","s","v"}) if(p[QString(k)+"_min"].toDouble()>p[QString(k)+"_max"].toDouble()) fail("HSV min must be <= max"); }
  if(d.type=="canny"&&p["low"].toDouble()>p["high"].toDouble()) fail("Canny low must be <= high");
  if(d.type=="components"&&(p["min_area"].toDouble()>p["max_area"].toDouble()||p["min_aspect"].toDouble()>p["max_aspect"].toDouble())) fail("Component min must be <= max");
  if(!s["enabled"].toBool()) continue; enabled=true;
  const QString t=d.type;
  if(t=="edge_localize") { if(mode!=1||geometry)fail("edge_localize must follow the final binary mask, once");geometry=1;continue; }
  if(t=="subpixel") { if(geometry!=1)fail("subpixel must immediately follow enabled edge_localize, once");geometry=2;continue; }
  if(geometry)fail("Place all mask processing before edge_localize / subpixel");
  if(t=="illumination"||t=="guided"||t=="sauvola") { if(mode!=0)fail(t+" requires a grayscale/channel intensity image, before binarization");mode=t=="sauvola"?1:0; }
  else if(t=="watershed") { if(mode!=1)fail("watershed requires a binary mask"); }
  else if(t=="gray") mode=mode==1?1:0;
  else if(t=="channel"||t=="hsv") { if(mode!=3) fail(t+" needs original/color BGR input; place before grayscale"); mode=t=="hsv"?1:0; }
  else if(t=="clahe"||t=="threshold"||t=="adaptive"||t=="canny") { if(mode==3) fail(t+" requires grayscale/channel step first"); mode=(t=="clahe")?0:1; }
  else if(t=="gaussian"||t=="median"||t=="bilateral") { if(mode==1) mode=0; }
  else if(t=="fill_holes"||t=="components") { if(mode!=1) fail(t+" requires a binary mask"); }
 }
 if(!enabled||mode!=1) fail("Final enabled step must produce a binary mask (threshold, HSV, adaptive or Canny)");
}
QJsonObject loadConfig(const QString& path) {
 QFile f(path); if(!f.open(QIODevice::ReadOnly)) fail(f.errorString());
 QJsonParseError e; auto d=QJsonDocument::fromJson(f.readAll(),&e);
 if(e.error!=QJsonParseError::NoError||!d.isObject()) fail("Invalid JSON: "+e.errorString());
 auto c=d.object(); validate(c); return c;
}
void saveJson(const QString& path,const QJsonObject& value) {
 QSaveFile f(path); if(!f.open(QIODevice::WriteOnly)) fail(f.errorString());
 auto bytes=QJsonDocument(value).toJson(); if(f.write(bytes)!=bytes.size()||!f.commit()) fail("Cannot save "+path);
}
cv::Mat readImage(const QString& path,bool mask) {
 QFile f(path); if(!f.open(QIODevice::ReadOnly)) fail("Cannot read "+path+": "+f.errorString()); auto b=f.readAll();
 std::vector<uchar> bytes(b.begin(),b.end()); auto m=cv::imdecode(bytes,mask?cv::IMREAD_UNCHANGED:cv::IMREAD_COLOR);
 if(m.empty()) fail("Cannot decode "+path);
 if(mask) { if(m.type()!=CV_8UC1) fail("Ground-truth mask must be single-channel 8-bit PNG"); cv::compare(m,0,m,cv::CMP_GT); }
 return m;
}
void writeImage(const QString& path,const cv::Mat& image) {
 std::vector<uchar> b; if(!cv::imencode(".png",image,b)) fail("PNG encode failed");
 QSaveFile f(path); if(!f.open(QIODevice::WriteOnly)) fail(f.errorString());
 if(f.write(reinterpret_cast<const char*>(b.data()),qint64(b.size()))!=qint64(b.size())||!f.commit()) fail("Cannot write "+path);
}
std::vector<cv::Rect> maskBoxes(const cv::Mat& mask) {
 cv::Mat labels,stats,centroids; int n=cv::connectedComponentsWithStats(mask,labels,stats,centroids,8);
 std::vector<cv::Rect> out; for(int i=1;i<n;i++) out.emplace_back(stats.at<int>(i,0),stats.at<int>(i,1),stats.at<int>(i,2),stats.at<int>(i,3)); return out;
}
Result run(const cv::Mat& input,const QJsonObject& c,bool capture,const std::function<bool()>& cancelled) {
 validate(c); if(input.empty()||input.type()!=CV_8UC3) fail("Input must be nonempty 8-bit BGR");
 Result r; cv::Mat m=input.clone(),intensity;cv::cvtColor(input,intensity,cv::COLOR_BGR2GRAY); QElapsedTimer total; total.start(); int index=0;
 for(const auto& v:c["steps"].toArray()) {
  if(cancelled&&cancelled()) fail("Cancelled"); ++index;
  auto s=v.toObject(); if(!s["enabled"].toBool()) continue; auto p=s["params"].toObject(); const auto t=s["type"].toString();
  auto n=[&](const char* k){return p[k].toInt();}; auto x=[&](const char* k){return p[k].toDouble();}; QElapsedTimer timer; timer.start();
  if(t=="edge_localize") {
   r.boundaries=localizeEdges(m,n("source")==1?intensity:input,n("radius"),x("sigma"),x("min_gradient"),n("normal_span"),cancelled);r.hasBoundaries=true;
  }
  else if(t=="subpixel") { refineSubpixel(r.boundaries,x("min_curvature"),cancelled);r.subpixel=true; }
  else if(t=="illumination") m=correctIllumination(m,x("sigma"),n("divide"));
  else if(t=="guided") m=guidedSmooth(m,n("radius"),x("epsilon"));
  else if(t=="sauvola") m=sauvolaMask(m,n("block"),x("k"),x("range"),n("inverse"));
  else if(t=="watershed") m=watershedSplit(m,input,n("peak_radius"),x("peak_fraction"));
  else if(t=="gray") { if(m.channels()==3) cv::cvtColor(m,m,cv::COLOR_BGR2GRAY); }
  else if(t=="channel") { if(n("space")==1) cv::cvtColor(m,m,cv::COLOR_BGR2HSV); if(n("space")==2) cv::cvtColor(m,m,cv::COLOR_BGR2Lab); cv::extractChannel(m,m,n("channel")); }
  else if(t=="clahe") cv::createCLAHE(x("clip"),cv::Size(n("grid"),n("grid")))->apply(m,m);
  else if(t=="gaussian") cv::GaussianBlur(m,m,cv::Size(n("kernel"),n("kernel")),x("sigma"));
  else if(t=="median") cv::medianBlur(m,m,n("kernel"));
  else if(t=="bilateral") { cv::Mat b; cv::bilateralFilter(m,b,n("diameter"),x("color"),x("space")); m=b; }
  else if(t=="threshold") cv::threshold(m,m,x("value"),255,(n("inverse")?cv::THRESH_BINARY_INV:cv::THRESH_BINARY)|(n("otsu")?cv::THRESH_OTSU:0));
  else if(t=="adaptive") cv::adaptiveThreshold(m,m,255,n("gaussian")?cv::ADAPTIVE_THRESH_GAUSSIAN_C:cv::ADAPTIVE_THRESH_MEAN_C,n("inverse")?cv::THRESH_BINARY_INV:cv::THRESH_BINARY,n("block"),x("c"));
  else if(t=="hsv") { cv::cvtColor(m,m,cv::COLOR_BGR2HSV); cv::inRange(m,cv::Scalar(n("h_min"),n("s_min"),n("v_min")),cv::Scalar(n("h_max"),n("s_max"),n("v_max")),m); }
  else if(t=="canny") cv::Canny(m,m,x("low"),x("high"));
  else if(t=="morphology") { static const int ops[]={cv::MORPH_OPEN,cv::MORPH_CLOSE,cv::MORPH_ERODE,cv::MORPH_DILATE,cv::MORPH_TOPHAT,cv::MORPH_BLACKHAT}; cv::morphologyEx(m,m,ops[n("operation")],cv::getStructuringElement(n("shape"),cv::Size(n("kernel"),n("kernel"))),cv::Point(-1,-1),n("iterations")); }
  else if(t=="invert") cv::bitwise_not(m,m);
  else if(t=="fill_holes") { cv::Mat padded; cv::copyMakeBorder(m,padded,1,1,1,1,cv::BORDER_CONSTANT,0); cv::floodFill(padded,cv::Point(0,0),255); cv::bitwise_not(padded,padded); cv::bitwise_or(m,padded(cv::Rect(1,1,m.cols,m.rows)),m); }
  else if(t=="components") {
   cv::Mat labels,stats,centroids; int count=cv::connectedComponentsWithStats(m,labels,stats,centroids,8); cv::Mat out=cv::Mat::zeros(m.size(),CV_8UC1); int largest=-1,areaMax=0;
   std::vector<uchar> keep(count,0);
   for(int i=1;i<count;i++) { int area=stats.at<int>(i,cv::CC_STAT_AREA),w=stats.at<int>(i,cv::CC_STAT_WIDTH),h=stats.at<int>(i,cv::CC_STAT_HEIGHT); double aspect=double(w)/h,extent=double(area)/(double(w)*h);
    if(area<x("min_area")||area>x("max_area")||aspect<x("min_aspect")||aspect>x("max_aspect")||extent<x("min_extent")) continue;
    if(n("largest_only")) { if(area>areaMax) {areaMax=area;largest=i;} } else keep[i]=255;
   }
   if(largest>=0) keep[largest]=255;
   for(int y=0;y<m.rows;y++){auto* label=labels.ptr<int>(y);auto* target=out.ptr<uchar>(y);for(int col=0;col<m.cols;col++)target[col]=keep[label[col]];}
   m=out;
  }
  const bool geometryStep=t=="edge_localize"||t=="subpixel";
  if(m.channels()==1&&(t=="gray"||t=="channel"||t=="clahe"||t=="gaussian"||t=="median"||t=="bilateral"||t=="illumination"||t=="guided"))intensity=m.clone();
  if(capture) {
   Stage stage;stage.name=QString::number(index)+". "+t;stage.stepIndex=index-1;
   stage.image=geometryStep?input:m.clone();stage.ms=timer.nsecsElapsed()/1e6;
   if(geometryStep)stage.boundaries=r.boundaries;
   r.stages.push_back(std::move(stage));
  }
 }
 cv::compare(m,0,r.mask,cv::CMP_GT); r.boxes=maskBoxes(r.mask); if(r.boxes.size()>2000) fail("Too many detected regions (limit 2000); add component filtering"); r.overlay=input.clone();
 cv::Mat tint=input.clone(); tint.setTo(cv::Scalar(60,220,70),r.mask); cv::addWeighted(input,0.65,tint,0.35,0,r.overlay);
 for(const auto& box:r.boxes) cv::rectangle(r.overlay,box,cv::Scalar(0,210,255),2);
 r.ms=total.nsecsElapsed()/1e6; return r;
}
Metrics fromCounts(double tp,double fp,double fn,double tn) {
 Metrics m; m.tp=tp;m.fp=fp;m.fn=fn;m.tn=tn;
 m.iou=tp+fp+fn?tp/(tp+fp+fn):1; m.dice=2*tp+fp+fn?2*tp/(2*tp+fp+fn):1;
 m.precision=tp+fp?tp/(tp+fp):(fn?0:1); m.recall=tp+fn?tp/(tp+fn):1; m.accuracy=tp+fp+fn+tn?(tp+tn)/(tp+fp+fn+tn):1;return m;
}
Metrics evaluateMask(const cv::Mat& a,const cv::Mat& b) {
 if(a.empty()||b.empty()||a.type()!=CV_8UC1||b.type()!=CV_8UC1||a.size()!=b.size()) fail("Mask dimensions/type differ; no automatic resizing allowed");
 cv::Mat p,g,both; cv::compare(a,0,p,cv::CMP_GT);cv::compare(b,0,g,cv::CMP_GT);cv::bitwise_and(p,g,both);
 double tp=cv::countNonZero(both),fp=cv::countNonZero(p)-tp,fn=cv::countNonZero(g)-tp;return fromCounts(tp,fp,fn,double(a.total())-tp-fp-fn);
}
static double boxIou(const cv::Rect& a,const cv::Rect& b) { auto r=a&b; double intersection=double(r.width)*r.height; return intersection/(double(a.width)*a.height+double(b.width)*b.height-intersection); }
DetectionMetrics evaluateBoxes(const std::vector<cv::Rect>& p,const std::vector<cv::Rect>& g,double threshold) {
 if(!std::isfinite(threshold)||threshold<0||threshold>1) fail("Invalid box IoU threshold");
 if(p.size()>2000||g.size()>2000) fail("Too many boxes for evaluation (limit 2000); refine component filtering");
 // Maximum-cardinality bipartite matching; never count one target twice.
 std::vector<std::vector<int>> edges(p.size());
 for(size_t i=0;i<p.size();i++) { for(size_t j=0;j<g.size();j++) if(boxIou(p[i],g[j])>=threshold&&boxIou(p[i],g[j])>0) edges[i].push_back(int(j));
  std::sort(edges[i].begin(),edges[i].end(),[&](int a,int b){return boxIou(p[i],g[a])>boxIou(p[i],g[b]);}); }
 std::vector<int> match(g.size(),-1);
 std::function<bool(int,std::vector<bool>&)> augment=[&](int i,std::vector<bool>& seen) { for(int j:edges[i]) if(!seen[j]) {seen[j]=true;if(match[j]<0||augment(match[j],seen)){match[j]=i;return true;}} return false; };
 DetectionMetrics m; for(size_t i=0;i<p.size();i++){std::vector<bool> seen(g.size());if(augment(int(i),seen))m.tp++;}
 for(size_t j=0;j<g.size();j++) if(match[j]>=0)m.meanMatchedIou+=boxIou(p[match[j]],g[j]);
 m.fp=int(p.size())-m.tp;m.fn=int(g.size())-m.tp;if(m.tp)m.meanMatchedIou/=m.tp;return m;
}
QJsonObject boxesJson(const std::vector<cv::Rect>& boxes) {
 QJsonArray a;for(const auto& b:boxes)a.append(QJsonArray{b.x,b.y,b.width,b.height});return {{"schema_version",1},{"boxes",a}};
}
std::vector<cv::Rect> readBoxes(const QString& path,cv::Size size) {
 QFile f(path);if(!f.open(QIODevice::ReadOnly))fail("Cannot read boxes: "+path);QJsonParseError e;auto d=QJsonDocument::fromJson(f.readAll(),&e);
 auto o=d.object();if(e.error!=QJsonParseError::NoError||!d.isObject()||o["schema_version"].toDouble()!=1||!o["boxes"].isArray())fail("Invalid boxes JSON: "+path);
 std::vector<cv::Rect> out;for(auto v:o["boxes"].toArray()){if(!v.isArray()||v.toArray().size()!=4)fail("Box must be [x,y,width,height]");auto a=v.toArray();for(auto n:a)if(!n.isDouble()||std::floor(n.toDouble())!=n.toDouble()||n.toDouble()<0||n.toDouble()>100000000)fail("Invalid box coordinate");cv::Rect r(a[0].toInt(),a[1].toInt(),a[2].toInt(),a[3].toInt());if(r.width<=0||r.height<=0||r.x>size.width-r.width||r.y>size.height-r.height)fail("Box outside image");out.push_back(r);}return out;
}
}
