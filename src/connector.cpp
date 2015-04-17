#include "./connector.h"
#include <string>

Connector::Connector(Eigen::Vector3f anchor, Eigen::Vector3f label)
  : anchor(anchor), label(label)
{
}

Connector::~Connector()
{
}

void Connector::initialize(Gl *gl)
{
  shaderProgram = std::unique_ptr<ShaderProgram>(
      new ShaderProgram(gl, ":shader/line.vert", ":shader/line.frag"));

  vertexArrayObject.create();
  vertexArrayObject.bind();

  shaderProgram->bind();

  float positions[6]{ anchor.x(), anchor.y(), anchor.z(),
                       label.x(),  label.y(),  label.z() };
  createBuffer(QOpenGLBuffer::Type::VertexBuffer, positions, "position", 3, 6);

  shaderProgram->release();
  vertexArrayObject.release();
}

void Connector::render(Gl *gl, const RenderData &renderData)
{
  if (!shaderProgram.get())
    initialize(gl);

  shaderProgram->bind();

  Eigen::Matrix4f modelViewProjection = renderData.projectionMatrix *
                                        renderData.viewMatrix;
  shaderProgram->setUniform("modelViewProjectionMatrix", modelViewProjection);
  shaderProgram->setUniform("color", color);

  vertexArrayObject.bind();

  gl->glLineWidth(3.0f);
  glAssert(gl->glDrawArrays(GL_LINES, 0, 2));

  vertexArrayObject.release();
  shaderProgram->release();
}

template <class ElementType>
void Connector::createBuffer(QOpenGLBuffer::Type bufferType, ElementType *data,
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

