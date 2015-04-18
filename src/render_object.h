#ifndef SRC_RENDER_OBJECT_H_

#define SRC_RENDER_OBJECT_H_

#include <memory>
#include <string>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include "./gl.h"
#include "./shader_program.h"

/**
 * \brief
 *
 *
 */
class RenderObject
{
 public:
  RenderObject(Gl *gl, std::string vertexShaderPath,
               std::string fragmentShaderPath);
  virtual ~RenderObject();

  template <class ElementType>
  void createBuffer(QOpenGLBuffer::Type bufferType, ElementType *data,
                    std::string usage, int perVertexElements,
                    int numberOfVertices)
  {
    QOpenGLBuffer buffer(bufferType);
    buffer.create();
    buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    buffer.bind();
    buffer.allocate(data,
                    numberOfVertices * perVertexElements * sizeof(ElementType));
    glCheckError();

    if (bufferType != QOpenGLBuffer::Type::IndexBuffer)
      shaderProgram->enableAndSetAttributes(usage, perVertexElements);

    buffers.push_back(buffer);
  }

  void bind();

  void release();

  void releaseBuffers();

  std::unique_ptr<ShaderProgram> shaderProgram;

 private:
  QOpenGLVertexArrayObject vertexArrayObject;
  std::vector<QOpenGLBuffer> buffers;
};

#endif  // SRC_RENDER_OBJECT_H_
