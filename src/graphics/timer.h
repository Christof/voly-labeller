#ifndef SRC_GRAPHICS_TIMER_H_

#define SRC_GRAPHICS_TIMER_H_

#include "./gl.h"

namespace Graphics
{

/**
 * \brief
 *
 *
 */
class Timer
{
 public:
  Timer();
  ~Timer();

  void initialize(Gl *gl);
  void start();
  void stop();
  double waitResult();

 private:
  GLint64 done();
  void checkTimerSupport();

  GLuint handle = 0;
  Gl *gl;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TIMER_H_
