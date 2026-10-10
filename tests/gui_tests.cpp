#include "window.h"
#include <QtWidgets>
#include <QtTest/QTest>
#include <opencv2/imgproc.hpp>
#include <iostream>
static void require(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
static void waitUntil(std::function<bool()> fn){QElapsedTimer timer;timer.start();while(!fn()&&timer.elapsed()<10000)QTest::qWait(20);require(fn(),"GUI preview timeout");}
int main(int argc,char** argv){QApplication app(argc,argv);QApplication::setStyle("Fusion");try{QTemporaryDir d;cv::Mat image(500,700,CV_8UC3,cv::Scalar(25,30,35));cv::rectangle(image,cv::Rect(180,130,340,210),cv::Scalar(190,195,200),cv::FILLED);auto path=d.filePath("synthetic-pad.png");pad::writeImage(path,image);Window w;w.show();w.openImage(path);auto* state=w.findChild<QLabel*>("previewState");auto* stages=w.findChild<QComboBox*>("stageSelector");auto* metrics=w.findChild<QLabel*>("metrics");require(state&&stages&&metrics,"GUI controls");waitUntil([&]{return state->text().startsWith("Preview ready");});require(stages->count()==8,"stage preview count");require(metrics->text().contains("1 regions"),"region displayed");stages->setCurrentIndex(2);QTest::qWait(50);stages->setCurrentIndex(0);
 // Toggle grayscale off: validation must clear old result, not show stale metrics.
 auto* list=w.findChild<QListWidget*>();require(list,"pipeline list");list->item(0)->setCheckState(Qt::Unchecked);waitUntil([&]{return state->text().startsWith("Invalid pipeline");});require(!metrics->text().contains("1 regions"),"stale score cleared");list->item(0)->setCheckState(Qt::Checked);waitUntil([&]{return state->text().startsWith("Preview ready");});
 // Exercise reorder through actual GUI buttons and verify both step types survive.
 QPushButton* down=nullptr;QPushButton* up=nullptr;for(auto* b:w.findChildren<QPushButton*>()){if(b->text()=="↓")down=b;if(b->text()=="↑")up=b;}require(down&&up,"reorder controls");list->setCurrentRow(0);QTest::mouseClick(down,Qt::LeftButton);require(list->item(0)->text().contains("gaussian")&&list->item(1)->text().contains("gray"),"reorder preserves distinct steps");QTest::mouseClick(up,Qt::LeftButton);waitUntil([&]{return state->text().startsWith("Preview ready");});QTest::qWait(100);
 auto* info=w.findChild<QLabel*>("stageInfo");auto* next=w.findChild<QPushButton*>("nextStep");require(info&&next,"step viewer controls");
 list->setCurrentRow(1);require(stages->currentData().toInt()==1&&info->text().contains("gaussian"),"click selects stage output");
 QTest::mouseClick(next,Qt::LeftButton);require(stages->currentData().toInt()==2&&info->text().contains("threshold"),"next step navigation");
 auto* action=w.findChild<QAction*>("precisionPreset");require(action,"precision preset");action->trigger();
 waitUntil([&]{return state->text().startsWith("Preview ready");});require(info->text().contains("subpixel")&&info->text().contains("refined"),"subpixel preview and stats");
 auto* view=w.findChild<ImageView*>("outputView");int paths=0;for(auto* item:view->scene()->items())if(item->type()==QGraphicsPathItem::Type)paths++;require(paths>0,"vector boundaries drawn");
 view->scale(2,2);double zoom=view->transform().m11();stages->setCurrentIndex(0);require(std::abs(view->transform().m11()-zoom)<1e-9,"zoom preserved across views");
 require(w.grab().save("gui-preview.png"),"screenshot saved");std::cout<<"GUI preview, validation, stage view and reorder passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
