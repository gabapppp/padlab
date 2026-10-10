#include "window.h"
#include <QtWidgets>
#include <QtConcurrent/QtConcurrentRun>
#include <QJsonArray>
#include <QJsonDocument>
#include <opencv2/imgproc.hpp>
#include <algorithm>
ImageView::ImageView(QWidget* p):QGraphicsView(p){setScene(new QGraphicsScene(this));setDragMode(ScrollHandDrag);setBackgroundBrush(QColor("#111827"));setTransformationAnchor(AnchorUnderMouse);setRenderHint(QPainter::SmoothPixmapTransform,false);}
void ImageView::setImage(const cv::Mat& m,const std::vector<pad::EdgeContour>& boundaries){
 const auto previous=scene()->sceneRect();scene()->clear();if(m.empty())return;
 cv::Mat rgb;if(m.channels()==1)cv::cvtColor(m,rgb,cv::COLOR_GRAY2RGB);else cv::cvtColor(m,rgb,cv::COLOR_BGR2RGB);
 QImage q(rgb.data,rgb.cols,rgb.rows,int(rgb.step),QImage::Format_RGB888);scene()->addPixmap(QPixmap::fromImage(q.copy()));
 for(const auto& contour:boundaries){
  if(contour.points.empty())continue;QPainterPath seed,path,rejected;
  auto qp=[](const cv::Point2f& p){return QPointF(p.x+0.5,p.y+0.5);};
  seed.moveTo(qp(contour.points.front().seed));path.moveTo(qp(contour.points.front().position));
  for(const auto& p:contour.points){seed.lineTo(qp(p.seed));path.lineTo(qp(p.position));if(!p.localized)rejected.addEllipse(qp(p.position),0.25,0.25);}
  seed.closeSubpath();path.closeSubpath();
  QPen reference(QColor("#f3f4f6"),1,Qt::DashLine);reference.setCosmetic(true);scene()->addPath(seed,reference);
  QPen edge(QColor("#00e5ff"),1.2);edge.setCosmetic(true);scene()->addPath(path,edge);
  QPen weak(QColor("#ff9800"),2);weak.setCosmetic(true);scene()->addPath(rejected,weak);
 }
 scene()->setSceneRect(0,0,m.cols,m.rows);if(previous!=scene()->sceneRect())fit();
}
void ImageView::fit(){if(!scene()->sceneRect().isEmpty())fitInView(scene()->sceneRect(),Qt::KeepAspectRatio);}
void ImageView::wheelEvent(QWheelEvent* e){double f=e->angleDelta().y()>0?1.2:1/1.2;double next=transform().m11()*f;if(next>0.01&&next<40)scale(f,f);e->accept();}
Window::Window(){
 setWindowTitle("PadLab — Image processing / OpenCV");resize(1460,900);
 auto* toolbar=addToolBar("Workspace");toolbar->setMovable(false);
 auto action=[&](const QString& label,std::function<void()> fn){auto* a=toolbar->addAction(label);connect(a,&QAction::triggered,this,[fn]{fn();});};
 action("Open image",[this]{auto p=QFileDialog::getOpenFileName(this,"Open image",{},"Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)");if(!p.isEmpty())openImage(p);});
 action("Load GT mask",[this]{auto p=QFileDialog::getOpenFileName(this,"Single-channel GT mask",{},"PNG (*.png)");if(!p.isEmpty()){maskPath_=p;schedule();}});
 action("Load GT boxes",[this]{auto p=QFileDialog::getOpenFileName(this,"GT boxes JSON",{},"JSON (*.json)");if(!p.isEmpty()){boxPath_=p;schedule();}});
 action("Clear GT",[this]{maskPath_.clear();boxPath_.clear();schedule();});toolbar->addSeparator();
 action("Load pipeline",[this]{auto p=QFileDialog::getOpenFileName(this,"Load pipeline",{},"JSON (*.json)");if(p.isEmpty())return;try{config_=pad::loadConfig(p);viewKey_=-1;rebuildSteps();schedule();}catch(const std::exception& e){error(e.what());}});
 action("Export pipeline",[this]{auto p=QFileDialog::getSaveFileName(this,"Export reusable pipeline", "pipeline.json","JSON (*.json)");if(p.isEmpty())return;try{pad::validate(config_);pad::saveJson(p,config_);statusBar()->showMessage("Pipeline exported. Use padlab-cli with the same JSON.",5000);}catch(const std::exception& e){error(e.what());}});
 action("Export result",[this]{exportResult();});
 auto* presets=new QToolButton;presets->setText("Presets");presets->setPopupMode(QToolButton::InstantPopup);auto* presetMenu=new QMenu(presets);
 auto* precision=presetMenu->addAction("Precision boundaries");precision->setObjectName("precisionPreset");
 auto* contrast=presetMenu->addAction("Low contrast + boundaries");
 connect(precision,&QAction::triggered,this,[this]{config_=pad::precisionConfig();rebuildSteps(config_["steps"].toArray().size()-1);viewKey_=steps_->currentRow();schedule();});
 connect(contrast,&QAction::triggered,this,[this]{config_=pad::precisionConfig(true);rebuildSteps(config_["steps"].toArray().size()-1);viewKey_=steps_->currentRow();schedule();});
 presets->setMenu(presetMenu);toolbar->addWidget(presets);
 action("Reset",[this]{config_=pad::defaultConfig();viewKey_=-1;rebuildSteps();schedule();});
 auto* root=new QWidget;setCentralWidget(root);auto* layout=new QVBoxLayout(root);fileLabel_=new QLabel("Open an image to begin. Data is processed locally.");layout->addWidget(fileLabel_);
 auto* split=new QSplitter;layout->addWidget(split,1);auto* left=new QWidget;auto* ll=new QVBoxLayout(left);ll->addWidget(new QLabel("PIPELINE · ordered steps"));steps_=new QListWidget;ll->addWidget(steps_,1);
 auto* addrow=new QHBoxLayout;addType_=new QComboBox;for(const auto& d:pad::definitions())addType_->addItem(d.label,d.type);addrow->addWidget(addType_,1);auto* add=new QPushButton("Add");addrow->addWidget(add);ll->addLayout(addrow);
 auto* commands=new QHBoxLayout;auto* up=new QPushButton("↑");auto* down=new QPushButton("↓");auto* remove=new QPushButton("Remove");commands->addWidget(up);commands->addWidget(down);commands->addWidget(remove);ll->addLayout(commands);
 auto* paramBox=new QGroupBox("Selected step parameters");params_=new QFormLayout(paramBox);auto* paramScroll=new QScrollArea;paramScroll->setWidgetResizable(true);paramScroll->setWidget(paramBox);ll->addWidget(paramScroll,1);
 auto* help=new QLabel("Click a step to inspect its output.\nColor space: 0 BGR, 1 HSV, 2 Lab\nMorphology: 0 open, 1 close, 2 erode,\n3 dilate, 4 top-hat, 5 black-hat\nShape: 0 rectangle, 1 cross, 2 ellipse\nHSV hue: 0–179. Kernels must be odd.\nEdge source: 0 original gray, 1 last intensity.\nBoundary coordinates are measurements;\nmask IoU uses the binary mask.");help->setWordWrap(true);ll->addWidget(help);split->addWidget(left);
 auto* tabs=new QTabWidget;split->addWidget(tabs);auto* previewPage=new QWidget;auto* pl=new QVBoxLayout(previewPage);
 auto* viewTools=new QHBoxLayout;stage_=new QComboBox;stage_->setObjectName("stageSelector");viewTools->addWidget(new QLabel("View:"));viewTools->addWidget(stage_,1);auto* fit=new QPushButton("Fit both");viewTools->addWidget(fit);pl->addLayout(viewTools);
 auto* inspectTools=new QHBoxLayout;auto* previousStep=new QPushButton("Previous step");previousStep->setObjectName("previousStep");auto* nextStep=new QPushButton("Next step");nextStep->setObjectName("nextStep");
 compareSteps_=new QCheckBox("Before / after step");compareSteps_->setObjectName("compareSteps");compareSteps_->setChecked(true);
 auto* exportStepImages=new QPushButton("Export all steps");inspectTools->addWidget(previousStep);inspectTools->addWidget(nextStep);inspectTools->addWidget(compareSteps_);inspectTools->addStretch();inspectTools->addWidget(exportStepImages);pl->addLayout(inspectTools);
 stageInfo_=new QLabel("Select a pipeline step to inspect its input and output.");stageInfo_->setObjectName("stageInfo");stageInfo_->setWordWrap(true);pl->addWidget(stageInfo_);
 auto navigate=[this](int direction){const auto& stages=last_.result.stages;if(stages.empty())return;int current=-1;for(int i=0;i<int(stages.size());i++)if(stages[i].stepIndex==viewKey_)current=i;int next=current<0?(direction>0?0:int(stages.size())-1):std::clamp(current+direction,0,int(stages.size())-1);steps_->setCurrentRow(stages[next].stepIndex);selectStepView();};
 connect(previousStep,&QPushButton::clicked,this,[navigate]{navigate(-1);});connect(nextStep,&QPushButton::clicked,this,[navigate]{navigate(1);});connect(compareSteps_,&QCheckBox::toggled,this,[this]{showStage();});connect(exportStepImages,&QPushButton::clicked,this,[this]{exportStages();});
 auto* views=new QSplitter;inputView_=new ImageView;inputView_->setObjectName("inputView");outputView_=new ImageView;outputView_->setObjectName("outputView");views->addWidget(inputView_);views->addWidget(outputView_);pl->addWidget(views,1);
 state_=new QLabel;state_->setObjectName("previewState");state_->setWordWrap(true);pl->addWidget(state_);metrics_=new QLabel("Metrics require a labeled image.");metrics_->setObjectName("metrics");metrics_->setWordWrap(true);metrics_->setTextInteractionFlags(Qt::TextSelectableByMouse);pl->addWidget(metrics_);
 auto* targets=new QHBoxLayout;maskTarget_=new QDoubleSpinBox;boxTarget_=new QDoubleSpinBox;for(auto* s:{maskTarget_,boxTarget_}){s->setRange(0,1);s->setDecimals(4);s->setSingleStep(0.001);}targets->addWidget(new QLabel("Mask IoU target (strict >)"));targets->addWidget(maskTarget_);targets->addWidget(new QLabel("Box match IoU (≥)"));targets->addWidget(boxTarget_);pl->addLayout(targets);
 tabs->addTab(previewPage,"Image / stages");
 auto* batchPage=new QWidget;auto* bl=new QVBoxLayout(batchPage);auto* folders=new QFormLayout;bl->addLayout(folders);
 auto folder=[&](const QString& name,bool output){auto* line=new QLineEdit;auto* row=new QWidget;auto* h=new QHBoxLayout(row);h->setContentsMargins(0,0,0,0);h->addWidget(line);auto* browse=new QPushButton("Browse");h->addWidget(browse);folders->addRow(name,row);connect(browse,&QPushButton::clicked,this,[this,line,output]{auto p=QFileDialog::getExistingDirectory(this,output?"Select parent for a new result directory":"Select directory");if(!p.isEmpty())line->setText(output?QDir(p).filePath("padlab-"+QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss-zzz")):p);});return line;};
 inputFolder_=folder("Images (non-recursive)",false);maskFolder_=folder("GT masks (optional)",false);boxFolder_=folder("GT boxes (optional)",false);outputFolder_=folder("New output directory",true);
 auto* note=new QLabel("Pairing: photo.jpg → masks/photo.jpg.png, boxes/photo.jpg.json.\nMask foreground = nonzero. Coordinates and mask size must match the original image. Box GT takes precedence for detection metrics.");note->setWordWrap(true);bl->addWidget(note);
 auto* runrow=new QHBoxLayout;batchButton_=new QPushButton("Run folder");cancelButton_=new QPushButton("Cancel");cancelButton_->setEnabled(false);failOnly_=new QCheckBox("Only errors / below target / unlabeled");runrow->addWidget(batchButton_);runrow->addWidget(cancelButton_);runrow->addWidget(failOnly_);runrow->addStretch();bl->addLayout(runrow);progress_=new QProgressBar;progress_->setValue(0);bl->addWidget(progress_);
 table_=new QTableWidget;table_->setColumnCount(8);table_->setHorizontalHeaderLabels({"Image","Status","Regions","ms","Mask IoU %","Dice %","Box TP/FP/FN","Error"});table_->setEditTriggers(QAbstractItemView::NoEditTriggers);table_->setSelectionBehavior(QAbstractItemView::SelectRows);table_->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);table_->horizontalHeader()->setStretchLastSection(true);bl->addWidget(table_,1);tabs->addTab(batchPage,"Folder evaluation");split->setSizes({340,1100});
 connect(add,&QPushButton::clicked,this,[this]{auto a=config_["steps"].toArray();a.append(pad::defaultStep(addType_->currentData().toString()));config_["steps"]=a;rebuildSteps(a.size()-1);viewKey_=steps_->currentRow();schedule();});
 auto move=[this](int direction){int i=steps_->currentRow();auto a=config_["steps"].toArray();int j=i+direction;if(i<0||j<0||j>=a.size())return;QJsonValue v=a[i];a[i]=a[j];a[j]=v;config_["steps"]=a;rebuildSteps(j);viewKey_=j;schedule();};
 connect(up,&QPushButton::clicked,this,[move]{move(-1);});connect(down,&QPushButton::clicked,this,[move]{move(1);});
 connect(remove,&QPushButton::clicked,this,[this]{int i=steps_->currentRow();if(i<0)return;auto a=config_["steps"].toArray();a.removeAt(i);config_["steps"]=a;rebuildSteps(qMax(0,i-1));viewKey_=steps_->currentRow();schedule();});
 connect(steps_,&QListWidget::currentRowChanged,this,[this]{if(!rebuilding_){buildParams();selectStepView();}});
 connect(steps_,&QListWidget::itemChanged,this,[this](QListWidgetItem* item){if(rebuilding_)return;int i=steps_->row(item);auto a=config_["steps"].toArray();auto s=a[i].toObject();s["enabled"]=item->checkState()==Qt::Checked;a[i]=s;config_["steps"]=a;schedule();});
 for(auto* s:{maskTarget_,boxTarget_})connect(s,QOverload<double>::of(&QDoubleSpinBox::valueChanged),this,[this]{if(rebuilding_)return;config_["evaluation"]=QJsonObject{{"mask_iou_target",maskTarget_->value()},{"box_iou_threshold",boxTarget_->value()}};schedule();});
 connect(fit,&QPushButton::clicked,this,[this]{inputView_->fit();outputView_->fit();});connect(stage_,QOverload<int>::of(&QComboBox::currentIndexChanged),this,[this]{if(stage_->currentIndex()>=0)viewKey_=stage_->currentData().toInt();showStage();});
 debounce_.setSingleShot(true);debounce_.setInterval(250);connect(&debounce_,&QTimer::timeout,this,[this]{preview();});
 connect(&previewWatcher_,&QFutureWatcher<Preview>::finished,this,[this]{auto r=previewWatcher_.result();if(r.generation==generation_){last_=std::move(r);if(!last_.error.isEmpty()){state_->setText("Error: "+last_.error);metrics_->clear();outputView_->setImage({});stage_->clear();}else{state_->setText("Preview ready · wheel to zoom, drag to pan · input on left");metrics_->setText("Final pipeline · "+pad::metricsText(last_.row));refreshViews();}}if(pending_){pending_=false;debounce_.start();}});
 connect(batchButton_,&QPushButton::clicked,this,[this]{startBatch();});connect(cancelButton_,&QPushButton::clicked,this,[this]{if(cancel_)cancel_->store(true);cancelButton_->setEnabled(false);});connect(failOnly_,&QCheckBox::toggled,this,[this]{fillTable();});
 connect(&batchWatcher_,&QFutureWatcher<BatchReply>::finished,this,[this]{batchButton_->setEnabled(true);cancelButton_->setEnabled(false);auto reply=batchWatcher_.result();if(!reply.error.isEmpty()){error(reply.error);return;}batch_=std::move(reply.batch);fillTable();statusBar()->showMessage(QString("%1 · %2 processed · %3 errors · %4 passed · report: %5").arg(batch_.cancelled?"Cancelled":"Complete").arg(batch_.rows.size()).arg(batch_.summary["errors"].toInt()).arg(batch_.summary["passed"].toInt()).arg(batchOutput_),0);});
 connect(table_,&QTableWidget::cellDoubleClicked,this,[this,tabs](int row,int){auto* item=table_->item(row,0);if(!item)return;QString p=item->data(Qt::UserRole).toString();config_=batchConfig_;rebuildSteps();openImage(p);maskPath_=item->data(Qt::UserRole+1).toString();boxPath_=item->data(Qt::UserRole+2).toString();schedule();tabs->setCurrentIndex(0);});
 setStyleSheet("QMainWindow { background: #f3f5f8; } QToolBar { background: #ffffff; border-bottom: 1px solid #d8dee8; spacing: 6px; padding: 6px; } QPushButton { padding: 6px 12px; } QGroupBox { font-weight: 600; } QListWidget::item { padding: 7px 3px; } QTabBar::tab { padding: 9px 18px; } QHeaderView::section { background: #e8edf4; padding: 7px; border: 0; } QTableWidget { gridline-color: #e1e6ee; } ");
 rebuildSteps();
}
Window::~Window(){if(cancel_)cancel_->store(true);previewWatcher_.waitForFinished();batchWatcher_.waitForFinished();}
void Window::error(const QString& message){QMessageBox::warning(this,"PadLab",message);}
void Window::rebuildSteps(int selected){rebuilding_=true;steps_->clear();auto a=config_["steps"].toArray();for(int i=0;i<a.size();i++){auto s=a[i].toObject();auto* item=new QListWidgetItem(QString::number(i+1)+". "+s["type"].toString(),steps_);item->setFlags(item->flags()|Qt::ItemIsUserCheckable);item->setCheckState(s["enabled"].toBool()?Qt::Checked:Qt::Unchecked);}steps_->setCurrentRow(qMin(selected,a.size()-1));auto e=config_["evaluation"].toObject();maskTarget_->setValue(e["mask_iou_target"].toDouble());boxTarget_->setValue(e["box_iou_threshold"].toDouble());rebuilding_=false;buildParams();}
void Window::buildParams(){while(params_->count()){auto* item=params_->takeAt(0);if(item->widget())item->widget()->deleteLater();delete item;}int i=steps_->currentRow();if(i<0)return;auto s=config_["steps"].toArray()[i].toObject();for(const auto& d:pad::definitions())if(d.type==s["type"].toString())for(const auto& p:d.parameters){auto* spin=new QDoubleSpinBox;spin->setRange(p.min,p.max);spin->setDecimals(p.integer?0:4);spin->setSingleStep((p.key=="kernel"||p.key=="block")?2:((p.key=="epsilon"||p.key=="min_curvature")?0.001:(p.integer?1:0.05)));spin->setValue(s["params"].toObject()[p.key].toDouble());params_->addRow(p.key,spin);connect(spin,QOverload<double>::of(&QDoubleSpinBox::valueChanged),this,[this,i,key=p.key](double value){auto a=config_["steps"].toArray();if(i>=a.size())return;auto step=a[i].toObject();auto p=step["params"].toObject();p[key]=value;step["params"]=p;a[i]=step;config_["steps"]=a;schedule();});}}
void Window::schedule(){generation_++;last_=Preview{};metrics_->setText("Configuration changed · awaiting preview");state_->clear();stageInfo_->setText("Recomputing selected view…");stage_->clear();outputView_->setImage({});debounce_.start();}
void Window::openImage(const QString& path){try{auto m=pad::readImage(path);image_=m;imagePath_=path;maskPath_.clear();boxPath_.clear();inputView_->setImage(m);fileLabel_->setText(path+QString(" · %1 × %2").arg(m.cols).arg(m.rows));schedule();}catch(const std::exception& e){error(e.what());}}
void Window::preview(){if(image_.empty())return;try{pad::validate(config_);}catch(const std::exception& e){state_->setText("Invalid pipeline: "+QString::fromUtf8(e.what()));return;}if(previewWatcher_.isRunning()){pending_=true;return;}auto image=image_.clone();auto c=config_;auto file=imagePath_,mask=maskPath_,boxes=boxPath_;auto generation=generation_;state_->setText("Processing…");previewWatcher_.setFuture(QtConcurrent::run([image,c,file,mask,boxes,generation]{Preview p;p.generation=generation;try{p.result=pad::run(image,c);p.row=pad::assess(file,p.result,c,mask,boxes);if(!mask.isEmpty()){p.truth=pad::readImage(mask,true);p.difference=image.clone();for(int y=0;y<image.rows;y++)for(int x=0;x<image.cols;x++){bool a=p.result.mask.at<uchar>(y,x),b=p.truth.at<uchar>(y,x);if(a||b)p.difference.at<cv::Vec3b>(y,x)=a&&b?cv::Vec3b(0,210,0):(a?cv::Vec3b(0,0,255):cv::Vec3b(220,0,220));}}}catch(const std::exception& e){p.error=QString::fromUtf8(e.what());}return p;}));}
void Window::refreshViews(){
 QSignalBlocker block(stage_);stage_->clear();
 stage_->addItem("Overlay + boxes",-1);stage_->addItem("Final mask",-2);stage_->addItem("Original image",-5);
 if(!last_.truth.empty()){stage_->addItem("Ground truth mask",-3);stage_->addItem("Pixel differences (green TP, red FP, magenta FN)",-4);}
 for(const auto& s:last_.result.stages)stage_->addItem(s.name+QString(" · %1 ms").arg(s.ms,0,'f',2),s.stepIndex);
 stage_->setCurrentIndex(stage_->findData(viewKey_));showStage();
}
void Window::selectStepView(){
 viewKey_=steps_->currentRow();if(viewKey_<0)return;
 {QSignalBlocker block(stage_);stage_->setCurrentIndex(stage_->findData(viewKey_));}
 showStage();
}
void Window::showStage(){
 if(last_.result.mask.empty())return;
 inputView_->setImage(image_);
 if(viewKey_<0){
  if(viewKey_==-1)outputView_->setImage(last_.result.overlay,last_.result.boundaries);
  if(viewKey_==-2)outputView_->setImage(last_.result.mask);
  if(viewKey_==-3)outputView_->setImage(last_.truth);
  if(viewKey_==-4)outputView_->setImage(last_.difference);
  if(viewKey_==-5)outputView_->setImage(image_);
  stageInfo_->setText("Original image on left · "+stage_->currentText()+" on right");return;
 }
 const auto& stages=last_.result.stages;
 for(size_t i=0;i<stages.size();i++)if(stages[i].stepIndex==viewKey_){
  const auto& s=stages[i];
  if(compareSteps_->isChecked()&&i>0)inputView_->setImage(stages[i-1].image,stages[i-1].boundaries);
  outputView_->setImage(s.image,s.boundaries);
  QString detail=QString("%1 · %2 × %3 · %4 ms · %5 → output").arg(s.name).arg(s.image.cols).arg(s.image.rows).arg(s.ms,0,'f',2).arg(compareSteps_->isChecked()?"previous output":"original image");
  if(!s.boundaries.empty()){
   int points=0,localized=0,refined=0;for(const auto& contour:s.boundaries)for(const auto& p:contour.points){points++;localized+=p.localized;refined+=p.refined;}
   detail+=QString("\nBoundary: %1 / %2 localized, %3 refined · cyan = measured, dashed = seed, orange = rejected").arg(localized).arg(points).arg(refined);
  }
  stageInfo_->setText(detail);return;
 }
 outputView_->setImage({});stageInfo_->setText("This step is disabled; it has no output. Enable it to inspect the result.");
}
void Window::exportStages(){
 if(last_.result.stages.empty()||last_.generation!=generation_||!last_.error.isEmpty()){error("Wait for a valid current preview first");return;}
 auto parent=QFileDialog::getExistingDirectory(this,"Export step images into a new subdirectory");if(parent.isEmpty())return;
 try{QString path=QDir(parent).filePath("padlab-steps-"+QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss-zzz"));
  pad::exportStageImages(path,image_,last_.result,config_);statusBar()->showMessage("Step images exported: "+path,8000);
 }catch(const std::exception& e){error(e.what());}
}
void Window::startBatch(){try{pad::validate(config_);if(inputFolder_->text().isEmpty()||outputFolder_->text().isEmpty())throw std::runtime_error("Select image and new output directories");auto files=pad::imageFiles(inputFolder_->text());if(files.isEmpty())throw std::runtime_error("No images in input folder");batchConfig_=config_;batchOutput_=outputFolder_->text();batch_.rows.clear();fillTable();cancel_=std::make_shared<std::atomic_bool>(false);auto cancel=cancel_;auto input=inputFolder_->text(),masks=maskFolder_->text(),boxes=boxFolder_->text(),output=batchOutput_;auto c=batchConfig_;progress_->setRange(0,files.size());progress_->setValue(0);batchButton_->setEnabled(false);cancelButton_->setEnabled(true);QPointer<Window> self(this);batchWatcher_.setFuture(QtConcurrent::run([self,input,masks,boxes,output,c,cancel]{BatchReply reply;try{reply.batch=pad::runFolder(input,masks,boxes,output,c,*cancel,[self](int value,int total){if(self)QMetaObject::invokeMethod(self,[self,value,total]{if(self){self->progress_->setRange(0,total);self->progress_->setValue(value);}},Qt::QueuedConnection);});}catch(const std::exception& e){reply.error=QString::fromUtf8(e.what());}return reply;}));}catch(const std::exception& e){error(e.what());}}
void Window::fillTable(){table_->setRowCount(0);for(const auto& r:batch_.rows){if(failOnly_->isChecked()&&r.status=="pass")continue;int i=table_->rowCount();table_->insertRow(i);QStringList values{QFileInfo(r.file).fileName(),r.status,QString::number(r.objects),QString::number(r.ms,'f',2),r.hasMask?QString::number(r.mask.iou*100,'f',4):"N/A",r.hasMask?QString::number(r.mask.dice*100,'f',4):"N/A",r.hasBoxes?QString("%1/%2/%3").arg(r.detection.tp).arg(r.detection.fp).arg(r.detection.fn):"N/A",r.error};for(int j=0;j<values.size();j++){auto* item=new QTableWidgetItem(values[j]);item->setToolTip(pad::metricsText(r));table_->setItem(i,j,item);}table_->item(i,0)->setData(Qt::UserRole,r.file);table_->item(i,0)->setData(Qt::UserRole+1,r.maskSource);table_->item(i,0)->setData(Qt::UserRole+2,r.boxSource=="derived from mask components"?QString():r.boxSource);}}
void Window::exportResult(){if(last_.result.mask.empty()||!last_.error.isEmpty()||last_.generation!=generation_){error("Wait for a valid current preview first");return;}auto parent=QFileDialog::getExistingDirectory(this,"Export result into a new subdirectory");if(parent.isEmpty())return;try{QString path=QDir(parent).filePath("padlab-image-"+QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss-zzz"));if(!QDir().mkdir(path))throw std::runtime_error("Cannot create result folder");QDir d(path);pad::writeImage(d.filePath("mask.png"),last_.result.mask);pad::writeImage(d.filePath("overlay.png"),pad::boundaryOverlay(last_.result.overlay,last_.result.boundaries));pad::saveJson(d.filePath("boundaries.json"),pad::boundariesJson(last_.result));pad::saveJson(d.filePath("boxes.json"),pad::boxesJson(last_.result.boxes));pad::saveJson(d.filePath("pipeline.json"),config_);pad::saveJson(d.filePath("metrics.json"),pad::rowJson(last_.row));for(size_t i=0;i<last_.result.boxes.size();i++){auto box=last_.result.boxes[i];pad::writeImage(d.filePath(QString("pad-%1.png").arg(i)),image_(box));pad::writeImage(d.filePath(QString("pad-%1-mask.png").arg(i)),last_.result.mask(box));}statusBar()->showMessage("Exported: "+path,8000);}catch(const std::exception& e){error(e.what());}}
