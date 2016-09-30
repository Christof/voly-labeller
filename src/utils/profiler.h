#ifndef SRC_UTILS_PROFILER_H_

#define SRC_UTILS_PROFILER_H_

#include <QLoggingCategory>
#include <QElapsedTimer>

/**
 * \brief Measures its lifetime and logs it in the destructor
 *
 * Pass the name of the profiled method or block and a QLoggingCategory to the
 * constructor. These are used for the output in the destructor.
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
