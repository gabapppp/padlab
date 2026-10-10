#pragma once
#include "dataset.h"
#include <QMainWindow>
#include <QFutureWatcher>
#include <QGraphicsView>
#include <QTimer>
#include <memory>
class QListWidget;class QComboBox;class QFormLayout;class QLabel;class QTableWidget;class QLineEdit;class QProgressBar;class QCheckBox;class QPushButton;class QDoubleSpinBox;
class ImageView : public QGraphicsView {
public: explicit ImageView(QWidget* parent=nullptr);void setImage(const cv::Mat& m,const std::vector<pad::EdgeContour>& boundaries={});void fit();
protected:void wheelEvent(QWheelEvent* e) override;
};
struct Preview { pad::Result result;pad::Row row;cv::Mat truth,difference;QString error;quint64 generation=0; };
struct BatchReply { pad::Batch batch;QString error; };
class Window : public QMainWindow {
public: Window();~Window();void openImage(const QString& path);
private:
 QJsonObject config_=pad::defaultConfig(),batchConfig_;QString imagePath_,maskPath_,boxPath_;cv::Mat image_;Preview last_;quint64 generation_=0;bool rebuilding_=false,pending_=false;int viewKey_=-1;
 QListWidget *steps_;QComboBox *addType_,*stage_;QFormLayout* params_;QLabel *metrics_,*state_,*fileLabel_;ImageView *inputView_,*outputView_;
 QTableWidget* table_;QLineEdit *inputFolder_,*maskFolder_,*boxFolder_,*outputFolder_;QProgressBar* progress_;QPushButton *batchButton_,*cancelButton_;QCheckBox *failOnly_,*compareSteps_;QLabel *stageInfo_;QDoubleSpinBox *boxTarget_,*maskTarget_;
 QFutureWatcher<Preview> previewWatcher_;QFutureWatcher<BatchReply> batchWatcher_;QTimer debounce_;std::shared_ptr<std::atomic_bool> cancel_;pad::Batch batch_;QString batchOutput_;
 void rebuildSteps(int selected=0);void buildParams();void schedule();void preview();void showStage();void selectStepView();void refreshViews();void exportStages();void startBatch();void fillTable();void error(const QString& message);void exportResult();
};
