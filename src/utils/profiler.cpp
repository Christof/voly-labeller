#include "./profiler.h"
#include "./profiling_statistics.h"

Profiler::Profiler(const char *name, const QLoggingCategory &channel,
                   ProfilingStatistics *profilingStatistics)
  : name(name), channel(channel), profilingStatistics(profilingStatistics)
{
#if PROFILE
  timer.start();
#endif
}

Profiler::~Profiler()
{
#if PROFILE
  float ms = 1e-6 * timer.nsecsElapsed();
  if (profilingStatistics)
  {
    profilingStatistics->addResult(name, ms);
  }
  else
  {
    qCInfo(channel) << name << "took\t" << ms << "ms";
  }
#endif
}
