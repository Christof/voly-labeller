#include "./profiler.h"

Profiler::Profiler(const char *name, const QLoggingCategory &channel)
  : name(name), channel(channel)
{
#if PROFILE
  timer.start();
#endif
}

Profiler::~Profiler()
{
#if PROFILE
  float ms = 1e-6 * timer.nsecsElapsed();
  qCInfo(channel) << name << "took\t" << ms << "ms";
#endif
}
