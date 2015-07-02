#ifndef SRC_GRAPHICS_GL_H_

#define SRC_GRAPHICS_GL_H_

#include <QOpenGLFunctions_4_3_Core>
#include <QtOpenGLExtensions>
#include <QSize>

#ifndef NDEBUG  // debug mode

#include <iostream>
#include <cassert>
#include <string>

class QOpenGLContext;
class QOpenGLPaintDevice;

namespace Graphics
{

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
    Graphics::glCheckErrorFunction(__FILE__, __LINE__);                        \
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
 public:
  Gl();
  ~Gl();

  void setSize(QSize size);

  void initialize(QOpenGLContext *context, QSize size);

  QOpenGLPaintDevice *paintDevice;
  QSize size;

  QOpenGLExtension_NV_shader_buffer_load *getShaderBufferLoad() const;
  QOpenGLExtension_EXT_direct_state_access *getDirectStateAccess() const;

 private:
  QOpenGLExtension_NV_shader_buffer_load *shaderBufferLoad;
  QOpenGLExtension_EXT_direct_state_access *directStateAccess;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_GL_H_
