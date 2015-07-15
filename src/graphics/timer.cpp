#include "./timer.h"
#include <cassert>

namespace Graphics
{

Timer::Timer()
{
}

Timer::~Timer()
{
  assert(handle != 0);
  glAssert(gl->glDeleteQueries(1, &handle));
}

void Timer::initialize(Gl *gl)
{
  this->gl = gl;

  if (handle)
    return;

  // check to make sure functionality is supported
  checkTimerSupport();

  glAssert(gl->glGenQueries(1, &handle));
}

void Timer::start()
{
  assert(handle != 0);
  glAssert(gl->glBeginQuery(GL_TIME_ELAPSED, handle));
}

void Timer::stop()
{
  assert(handle != 0);
  glAssert(gl->glEndQuery(GL_TIME_ELAPSED));
}

double Timer::waitResult()
{
  int count = 0;

  while (true)
  {
    GLint64 elapsedTime = done();

    if (elapsedTime >= 0)
      return static_cast<double>(elapsedTime) * 1e-6;

    count++;

    if (count > 5000)
      return -1.0;
  }
}

GLint64 Timer::done()
{
  assert(handle != 0);
  int available = 0;
  glAssert(
      gl->glGetQueryObjectiv(handle, GL_QUERY_RESULT_AVAILABLE, &available));

  if (!available)
    return -1;

  GLuint64 elapsedTime = 0;
  glAssert(gl->glGetQueryObjectui64v(handle, GL_QUERY_RESULT, &elapsedTime));
  return static_cast<GLint64>(elapsedTime);
}

void Timer::checkTimerSupport()
{
  GLint bitsSupported;
  gl->glGetQueryiv(GL_TIME_ELAPSED, GL_QUERY_COUNTER_BITS, &bitsSupported);

  if (bitsSupported == 0)
    throw std::runtime_error("Hardware does not support timers");
}

}  // namespace Graphics

