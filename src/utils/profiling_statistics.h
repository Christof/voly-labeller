#ifndef SRC_UTILS_PROFILING_STATISTICS_H_

#define SRC_UTILS_PROFILING_STATISTICS_H_

#include <QLoggingCategory>
#include <map>
#include <string>

/**
 * \brief Calculates statistics (for now only average) for Profiler instances
 *
 * The constructor of Profiler has an optional parameter for a
 * ProfilingStatistics pointer. If it is passed it calls #addResult in its
 * destructor instead of logging each measurement. In the ProfilingStatistics
 * destructor the average is calculated and logged.
 */
class ProfilingStatistics
{
 public:
  ProfilingStatistics(const char *name, const QLoggingCategory &channel);
  ~ProfilingStatistics();

  void addResult(std::string profilerName, float elapsedMilliseconds);

 private:
  struct ProfilerResults
  {
    float elapsedMillisecondsSum = 0;
    int counter = 0;

    void add(float elapsed)
    {
      counter++;
      elapsedMillisecondsSum += elapsed;
    }

    float average()
    {
      return elapsedMillisecondsSum / counter;
    }
  };
  const char *name;
  const QLoggingCategory &channel;
  std::map<std::string, ProfilerResults> profilersResults;
};

#endif  // SRC_UTILS_PROFILING_STATISTICS_H_
