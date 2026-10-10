#include "dataset.h"
#include <QCoreApplication>
#include <QCommandLineParser>
#include <QJsonDocument>
#include <QTextStream>
#include <csignal>
static volatile std::sig_atomic_t interrupted=0;
static void onSignal(int){interrupted=1;}
int main(int argc,char** argv){QCoreApplication app(argc,argv);QCoreApplication::setApplicationName("padlab-cli");QCommandLineParser p;p.setApplicationDescription("Offline OpenCV pad pipeline runner; mask + box metrics");p.addHelpOption();p.addOption({{"c","config"},"Pipeline JSON","file"});p.addOption({{"i","input"},"Input image folder (non-recursive)","directory"});p.addOption({{"o","output"},"NEW output directory (parent must exist)","directory"});p.addOption({"masks","Ground-truth mask folder","directory"});p.addOption({"boxes","Ground-truth box folder","directory"});p.addOption({"write-default","Write default config and exit","file"});p.process(app);QTextStream out(stdout),err(stderr);try{if(p.isSet("write-default")){pad::saveJson(p.value("write-default"),pad::defaultConfig());return 0;}for(const auto& key:{"config","input","output"})if(!p.isSet(key))throw std::runtime_error(QString("Required: --%1").arg(key).toStdString());std::signal(SIGINT,onSignal);std::atomic_bool cancel{false};auto result=pad::runFolder(p.value("input"),p.value("masks"),p.value("boxes"),p.value("output"),pad::loadConfig(p.value("config")),cancel,[&](int n,int total){out<<n<<"/"<<total<<"\n";out.flush();if(interrupted)cancel.store(true);});out<<QJsonDocument(result.summary).toJson();return result.cancelled?130:(result.summary["errors"].toInt()?2:0);}catch(const std::exception& e){err<<e.what()<<"\n";return 1;}}
