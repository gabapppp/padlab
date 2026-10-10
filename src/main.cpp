#include "window.h"
#include <QApplication>
int main(int argc,char** argv){QApplication app(argc,argv);QApplication::setApplicationName("PadLab");QApplication::setStyle("Fusion");Window w;w.show();if(argc>1)w.openImage(QString::fromLocal8Bit(argv[1]));return app.exec();}
