#ifndef SRC_QUAD_H_

#define SRC_QUAD_H_

#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <vector>
#include <memory>
#include <string>
#include "./shader_program.h"
#include "./render_data.h"

class Texture;
/**
 * \brief
 *
 *
 */
class Quad
{
 public:
  Quad();
  virtual ~Quad();

  void render(Gl *gl, const RenderData &renderData,
              std::shared_ptr<Texture> texture);

  void initialize(Gl *gl);

 private:
  QOpenGLVertexArrayObject vertexArrayObject;
  std::vector<QOpenGLBuffer> buffers;
  std::unique_ptr<ShaderProgram> shaderProgram;
  static const int indexCount = 6;
  template <class ElementType>
  void createBuffer(QOpenGLBuffer::Type bufferType, ElementType *data,
                    std::string usage, int perVertexElements,
                    int numberOfVertices);
};

#endif  // SRC_QUAD_H_
