#ifndef SRC_APPLICATION_H_

#define SRC_APPLICATION_H_

#include <QtGui/QGuiApplication>

/**
 * \brief
 *
 *
 */
class Application
{
 public:
  Application(int &argc, char **argv);
  virtual ~Application();

  int execute();

 private:
  QGuiApplication application;
};

#endif  // SRC_APPLICATION_H_
