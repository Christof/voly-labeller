#ifndef SRC_GRAPHICS_RENDER_OBJECT_H_

#define SRC_GRAPHICS_RENDER_OBJECT_H_

#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <memory>
#include <string>
#include <vector>
#include "./shader_program.h"

namespace Graphics
{

class Gl;

/**
 * \brief Encapsulates buffers and an associated shader program
 *
 * It provides helper functions to bind and release the
 * shader and the vertex array object as well as all buffers.
 */
class RenderObject
{
 public:
  RenderObject(Gl *gl, std::string vertexShaderPath,
               std::string fragmentShaderPath);
  virtual ~RenderObject();

  std::shared_ptr<ShaderProgram> shaderProgram;

 private:
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_RENDER_OBJECT_H_
