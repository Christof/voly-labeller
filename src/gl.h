#ifndef SRC_GL_H_

#define SRC_GL_H_

#include <QOpenGLFunctions_4_3_Core>
#include <QtOpenGLExtensions>
#include <QSize>

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

class QOpenGLContext;
class QOpenGLPaintDevice;

/**
 * \brief Provides access to OpenGL functions
 *
 * This is used instead of directly using the Qt type to make
 * changing the OpenGl version easy
 */
class Gl : public QOpenGLFunctions_4_3_Core
{
 public:
  Gl();
  ~Gl();

  void setSize(QSize size);

  void initialize(QOpenGLContext *context, QSize size);

  QOpenGLPaintDevice *paintDevice;
  QSize size;

  const QOpenGLExtension_NV_shader_buffer_load *getShaderBufferLoad() const;

 private:
  QOpenGLExtension_NV_shader_buffer_load *shaderBufferLoad;
};

#endif  // SRC_GL_H_
