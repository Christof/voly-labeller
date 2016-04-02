#ifndef SRC_UTILS_LOGGING_H_

#define SRC_UTILS_LOGGING_H_

#include <QLoggingCategory>
#include <chrono>

inline float
calculateDurationSince(std::chrono::high_resolution_clock::time_point startTime)
{
  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> diff = endTime - startTime;

  return diff.count();
}

#endif  // SRC_UTILS_LOGGING_H_
