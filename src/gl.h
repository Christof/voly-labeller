#ifndef SRC_GL_H_

#define SRC_GL_H_

#include <QOpenGLFunctions_4_3_Core>

#ifndef NDEBUG  // debug mode

#include <iostream>
#include <cassert>
#include <string>

#ifndef __TO_STR
#define __TO_STR(x) __EVAL_STR(x)
#define __EVAL_STR(x) #x
#endif

inline void glCheckErrorFunction(std::string file, int line)
{
  GLuint err = glGetError();
  if (err != GL_NO_ERROR)
  {
    std::cerr << "OpenGL error(" << file << ":" << std::to_string(line)
              << ") : "
              << "code(" << err << ")" << std::endl;
    assert(false);
  }
}

#define glAssert(code)                                                         \
  code;                                                                        \
  {                                                                            \
    glCheckErrorFunction(__FILE__, __LINE__);                                  \
  }

#define glCheckError() glCheckErrorFunction(__FILE__, __LINE__)

#else  // No debug
#define glAssert(code) code;
#define glCheckError()
#endif

/**
 * \brief Provides access to OpenGL functions
 *
 * This is used instead of directly using the Qt type to make
 * changing the OpenGl version easy
 */
class Gl : public QOpenGLFunctions_4_3_Core
{
};

#endif  // SRC_GL_H_
