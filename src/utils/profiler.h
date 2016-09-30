#ifndef SRC_UTILS_PROFILER_H_

#define SRC_UTILS_PROFILER_H_

#include <QLoggingCategory>
#include <QElapsedTimer>

/**
 * \brief
 *
 *
 */
class Profiler
{
 public:
  Profiler(const char *name, const QLoggingCategory &channel);
  virtual ~Profiler();

 private:
  const char *name;
  const QLoggingCategory &channel;
  QElapsedTimer timer;
};

#endif  // SRC_UTILS_PROFILER_H_
