#include <QtGui/QGuiApplication>
#include "window.h"

int main(int argc, char **argv)
{
  QGuiApplication application(argc, argv);
  Window window;

  window.show();

  return application.exec();
}
