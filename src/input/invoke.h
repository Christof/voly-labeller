#ifndef SRC_INPUT_INVOKE_H_

#define SRC_INPUT_INVOKE_H_

#include <QString>

/**
 * \brief
 *
 *
 */
class Invoke
{
 public:
  Invoke(const Invoke &) = default;
  Invoke(QString targetType, QString source)
    : targetType(targetType), source(source)
  {
  }

  QString targetType;
  QString source;

 private:
  /* data */
};

#endif  // SRC_INPUT_INVOKE_H_
