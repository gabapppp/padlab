#include "dataset.h"
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QSaveFile>
#include <stdexcept>
namespace pad {
static void fail(const QString& s){throw std::runtime_error(s.toStdString());}
QStringList imageFiles(const QString& folder) {
 QDir d(folder);if(!d.exists())fail("Input directory does not exist");QStringList out;
 for(const auto& f:d.entryInfoList(QDir::Files,QDir::Name)){QString ext=f.suffix().toLower();if(QStringList{"png","jpg","jpeg","bmp","tif","tiff","webp"}.contains(ext))out<<f.absoluteFilePath();}return out;
}
Row assess(const QString& file,const Result& r,const QJsonObject& c,const QString& maskPath,const QString& boxPath) {
 Row row;row.file=file;row.objects=int(r.boxes.size());row.ms=r.ms;row.status="unlabeled";
 std::vector<cv::Rect> gt;
 if(!maskPath.isEmpty()&&QFileInfo::exists(maskPath)){auto m=readImage(maskPath,true);row.mask=evaluateMask(r.mask,m);row.hasMask=true;row.maskSource=maskPath;gt=maskBoxes(m);}
 if(!boxPath.isEmpty()&&QFileInfo::exists(boxPath)){gt=readBoxes(boxPath,r.mask.size());row.hasBoxes=true;row.boxSource=boxPath;}
 else if(row.hasMask){row.hasBoxes=true;row.boxSource="derived from mask components";}
 if(row.hasBoxes)row.detection=evaluateBoxes(r.boxes,gt,c["evaluation"].toObject()["box_iou_threshold"].toDouble());
 row.passed=(!row.hasMask||row.mask.iou>c["evaluation"].toObject()["mask_iou_target"].toDouble())&&(!row.hasBoxes||(row.detection.fp==0&&row.detection.fn==0));
 if(row.hasMask||row.hasBoxes)row.status=row.passed?"pass":"below_target";else row.passed=false;return row;
}
QString metricsText(const Row& r){
 QString s=QString("%1 regions | %2 ms | %3").arg(r.objects).arg(r.ms,0,'f',2).arg(r.status);
 if(r.hasMask)s+=QString("\nMask IoU %1% | Dice %2% | P %3% | R %4% | accuracy %5% (background included)").arg(r.mask.iou*100,0,'f',4).arg(r.mask.dice*100,0,'f',4).arg(r.mask.precision*100,0,'f',3).arg(r.mask.recall*100,0,'f',3).arg(r.mask.accuracy*100,0,'f',3);
 if(r.hasBoxes){auto m=fromCounts(r.detection.tp,r.detection.fp,r.detection.fn,0);s+=QString("\nBoxes TP %1 / FP %2 / FN %3 | P %4% | R %5%").arg(r.detection.tp).arg(r.detection.fp).arg(r.detection.fn).arg(m.precision*100,0,'f',3).arg(m.recall*100,0,'f',3);}
 if(!r.error.isEmpty())s+="\n"+r.error;return s;
}
QJsonObject rowJson(const Row& r){
 QJsonObject o{{"file",r.file},{"status",r.status},{"error",r.error},{"regions",r.objects},{"ms",r.ms},{"mask_source",r.maskSource},{"box_source",r.boxSource}};
 o["mask"]=r.hasMask?QJsonValue(QJsonObject{{"iou",r.mask.iou},{"dice",r.mask.dice},{"precision",r.mask.precision},{"recall",r.mask.recall},{"accuracy",r.mask.accuracy},{"tp",r.mask.tp},{"fp",r.mask.fp},{"fn",r.mask.fn},{"tn",r.mask.tn}}):QJsonValue(QJsonValue::Null);
 auto m=fromCounts(r.detection.tp,r.detection.fp,r.detection.fn,0);
 o["boxes"]=r.hasBoxes?QJsonValue(QJsonObject{{"tp",r.detection.tp},{"fp",r.detection.fp},{"fn",r.detection.fn},{"precision",m.precision},{"recall",m.recall},{"f1",m.dice},{"mean_matched_iou",r.detection.meanMatchedIou}}):QJsonValue(QJsonValue::Null);return o;
}
void exportStageImages(const QString& directory,const cv::Mat& original,const Result& result,const QJsonObject& config){
 if(QFileInfo::exists(directory)||!QDir().mkdir(directory))fail("Choose a new step export directory with an existing parent");
 QDir out(directory);writeImage(out.filePath("00-original.png"),original);saveJson(out.filePath("pipeline.json"),config);
 QJsonArray manifest;
 for(const auto& stage:result.stages){
  const QString file=QString("%1-%2.png").arg(stage.stepIndex+1,2,10,QChar('0')).arg(config["steps"].toArray()[stage.stepIndex].toObject()["type"].toString());
  writeImage(out.filePath(file),boundaryOverlay(stage.image,stage.boundaries));
  manifest.append(QJsonObject{{"step_index",stage.stepIndex},{"name",stage.name},{"file",file},{"milliseconds",stage.ms}});
 }
 saveJson(out.filePath("stages.json"),QJsonObject{{"schema_version",1},{"steps",manifest}});
 saveJson(out.filePath("boundaries.json"),boundariesJson(result));
}
static QString csv(QString s){s.replace("\"","\"\"");if(!s.isEmpty()&&QString("=+-@\t\r").contains(s[0]))s.prepend("'");return "\""+s+"\"";}
static bool inside(const QString& a,const QString& b){return a==b||a.startsWith(b+QDir::separator());}
Batch runFolder(const QString& input,const QString& masks,const QString& boxes,const QString& output,const QJsonObject& c,std::atomic_bool& cancel,const std::function<void(int,int)>& progress){
 validate(c);auto files=imageFiles(input);if(files.isEmpty())fail("No supported images in folder");
 if(!masks.isEmpty()&&!QDir(masks).exists())fail("Mask folder does not exist");if(!boxes.isEmpty()&&!QDir(boxes).exists())fail("Box folder does not exist");
 QString dest=QDir::cleanPath(QFileInfo(output).absoluteFilePath());
 if(QFileInfo::exists(dest))fail("Output must be a NEW directory; existing files will not be overwritten");
 QString parent=QFileInfo(dest).dir().canonicalPath();if(parent.isEmpty())fail("Output parent must exist");dest=QDir(parent).filePath(QFileInfo(dest).fileName());
 for(const auto& dir:{input,masks,boxes})if(!dir.isEmpty()){QString source=QDir(dir).canonicalPath();if(inside(dest,source)||inside(source,dest))fail("Output must be separate from image/ground-truth folders");}
 if(!QDir().mkdir(dest))fail("Cannot create output folder");QDir out(dest);out.mkdir("masks");out.mkdir("overlays");out.mkdir("boxes");out.mkdir("crops");out.mkdir("boundaries");saveJson(out.filePath("pipeline.json"),c);
 Batch batch;double tp=0,fp=0,fn=0,tn=0,macro=0;int labeled=0,boxLabeled=0,dtp=0,dfp=0,dfn=0,passed=0,errors=0;
 QString report="file,status,error,regions,ms,mask_iou,dice,pixel_precision,pixel_recall,pixel_accuracy,box_tp,box_fp,box_fn,box_mean_matched_iou,mask_source,box_source\n";
 for(int i=0;i<files.size();i++) {
  if(cancel.load()){batch.cancelled=true;break;}Row row;row.file=files[i];QString name=QFileInfo(files[i]).fileName();
  try {auto original=readImage(files[i]);auto result=run(original,c,false,[&]{return cancel.load();});if(cancel.load()){batch.cancelled=true;break;}
   row.objects=int(result.boxes.size());row.ms=result.ms;
   writeImage(out.filePath("masks/"+name+".png"),result.mask);writeImage(out.filePath("overlays/"+name+".png"),boundaryOverlay(result.overlay,result.boundaries));saveJson(out.filePath("boundaries/"+name+".json"),boundariesJson(result));saveJson(out.filePath("boxes/"+name+".json"),boxesJson(result.boxes));
   for(size_t k=0;k<result.boxes.size();k++){auto box=result.boxes[k];writeImage(out.filePath("crops/"+name+QString(".pad-%1.png").arg(k)),original(box));writeImage(out.filePath("crops/"+name+QString(".pad-%1-mask.png").arg(k)),result.mask(box));}
   row.maskSource=masks.isEmpty()?QString():QDir(masks).filePath(name+".png");row.boxSource=boxes.isEmpty()?QString():QDir(boxes).filePath(name+".json");
   row=assess(files[i],result,c,masks.isEmpty()?QString():QDir(masks).filePath(name+".png"),boxes.isEmpty()?QString():QDir(boxes).filePath(name+".json"));
   if(row.hasMask){labeled++;tp+=row.mask.tp;fp+=row.mask.fp;fn+=row.mask.fn;tn+=row.mask.tn;macro+=row.mask.iou;}
   if(row.hasBoxes){boxLabeled++;dtp+=row.detection.tp;dfp+=row.detection.fp;dfn+=row.detection.fn;}if(row.passed)passed++;
  } catch(const std::exception& e){if(cancel.load()){batch.cancelled=true;break;}row.hasMask=false;row.hasBoxes=false;row.status="error";row.error=QString::fromUtf8(e.what());errors++;}
  QStringList fields{csv(name),csv(row.status),csv(row.error),QString::number(row.objects),QString::number(row.ms,'f',3)};
  for(double v:{row.mask.iou,row.mask.dice,row.mask.precision,row.mask.recall,row.mask.accuracy})fields<<(row.hasMask?QString::number(v,'g',12):QString());
  for(int v:{row.detection.tp,row.detection.fp,row.detection.fn})fields<<(row.hasBoxes?QString::number(v):QString());
  fields<<(row.hasBoxes?QString::number(row.detection.meanMatchedIou,'g',12):QString())<<csv(row.maskSource)<<csv(row.boxSource);report+=fields.join(",")+"\n";batch.rows.push_back(row);if(progress)progress(i+1,files.size());
 }
 auto micro=fromCounts(tp,fp,fn,tn);auto dm=fromCounts(dtp,dfp,dfn,0);
 batch.summary={{"schema_version",1},{"cancelled",batch.cancelled},{"input_images",files.size()},{"processed",int(batch.rows.size())},{"errors",errors},{"mask_labeled",labeled},{"box_labeled",boxLabeled},{"passed",passed},{"empty_mask_policy","both empty => IoU/Dice 1; metrics use nonzero foreground"},{"box_matching","maximum cardinality, one-to-one, IoU >= threshold"},{"mask_pass_policy","IoU strictly greater than target"},{"mask_macro_iou",labeled?QJsonValue(macro/labeled):QJsonValue(QJsonValue::Null)},{"mask_micro_iou",labeled?QJsonValue(micro.iou):QJsonValue(QJsonValue::Null)},{"mask_micro_dice",labeled?QJsonValue(micro.dice):QJsonValue(QJsonValue::Null)},{"box_precision",boxLabeled?QJsonValue(dm.precision):QJsonValue(QJsonValue::Null)},{"box_recall",boxLabeled?QJsonValue(dm.recall):QJsonValue(QJsonValue::Null)},{"box_f1",boxLabeled?QJsonValue(dm.dice):QJsonValue(QJsonValue::Null)},{"box_tp",dtp},{"box_fp",dfp},{"box_fn",dfn},{"evaluation",c["evaluation"]}};
 QSaveFile f(out.filePath("report.csv"));if(!f.open(QIODevice::WriteOnly))fail(f.errorString());auto bytes=report.toUtf8();if(f.write(bytes)!=bytes.size()||!f.commit())fail("Cannot save report");saveJson(out.filePath("summary.json"),batch.summary);QJsonArray rows;for(const auto& row:batch.rows)rows.append(rowJson(row));saveJson(out.filePath("metrics.json"),QJsonObject{{"images",rows}});return batch;
}
}
