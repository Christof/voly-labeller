#include "./profiling_statistics.h"
#include <string>

ProfilingStatistics::ProfilingStatistics(const char *name,
                                         const QLoggingCategory &channel)
  : name(name), channel(channel)
{
}

ProfilingStatistics::~ProfilingStatistics()
{
  qCInfo(channel) << "Profiling results for" << name;
  for (auto &pair : profilersResults)
  {
    qCInfo(channel) << "\t" << pair.first.c_str() << "took on average"
                    << pair.second.average() << "ms";
  }
}

void ProfilingStatistics::addResult(std::string profilerName,
                                    float elapsedMilliseconds)
{
  profilersResults[profilerName].add(elapsedMilliseconds);
}
