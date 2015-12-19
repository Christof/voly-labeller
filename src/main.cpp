#include <QDebug>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "./utils/cuda_helper.h"
#include "./application.h"

/**
 * \brief Setup logging
 *
 * [Documentation for the
 * pattern](http://doc.qt.io/qt-5/qtglobal.html#qSetMessagePattern)
 *
 * [Documentation for console formatting]
 * (https://en.wikipedia.org/wiki/ANSI_escape_code#Colors)
 */
void setupLogging()
{
  qputenv("QT_MESSAGE_PATTERN",
          QString("%{time [yyyy'-'MM'-'dd' "
                  "'hh':'mm':'ss]} "
                  "%{if-fatal}\033[31;1m%{endif}"
                  "%{if-critical}\033[31m%{endif}"
                  "%{if-warning}\033[33m%{endif}"
                  "%{if-info}\033[34m%{endif}"
                  "- %{threadid} "
                  "%{if-category}%{category}: %{endif}%{message}"
                  "%{if-warning}\n\t%{file}:%{line}\n\t%{backtrace depth=3 "
                  "separator=\"\n\t\"}%{endif}"
                  "%{if-critical}\n\t%{file}:%{line}\n\t%{backtrace depth=3 "
                  "separator=\"\n\t\"}%{endif}\033[0m").toUtf8());
  if (qgetenv("QT_LOGGING_CONF").size() == 0)
    qputenv("QT_LOGGING_CONF", "../config/logging.ini");
}

void setupCuda()
{
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0)
  {
    qCritical() << "No cuda device found!";
    exit(EXIT_FAILURE);
  }

  cudaDeviceProp prop;
  int device;
  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 3;
  prop.minor = 0;
  HANDLE_ERROR(cudaChooseDevice(&device, &prop));
  HANDLE_ERROR(cudaGLSetGLDevice(device));
}

int main(int argc, char **argv)
{
  setupLogging();

  setupCuda();

  Application application(argc, argv);

  return application.execute();
}

