#include "./quad.h"
#include <string>
#include "./gl.h"
#include "./texture.h"

Quad::Quad()
{
}

Quad::~Quad()
{
}

void Quad::initialize(Gl *gl)
{
  shaderProgram = std::unique_ptr<ShaderProgram>(
      new ShaderProgram(gl, ":shader/label.vert", ":shader/label.frag"));

  vertexArrayObject.create();
  vertexArrayObject.bind();

  shaderProgram->bind();

  float positions[12]{ 1.0f, 1.0f,  0.0f, -1.0f, 1.0f,  0.0f,
                       1.0f, -1.0f, 0.0f, -1.0f, -1.0f, 0.0f };
  createBuffer(QOpenGLBuffer::Type::VertexBuffer, positions, "position", 3, 12);
  float texcoords[8]{ 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f };
  createBuffer(QOpenGLBuffer::Type::VertexBuffer, texcoords, "texcoord", 2, 12);

  shaderProgram->release();
  vertexArrayObject.release();
}

void Quad::render(Gl *gl, const RenderData &renderData,
                  std::shared_ptr<Texture> texture)
{
  if (!shaderProgram.get())
    initialize(gl);

  shaderProgram->bind();

  Eigen::Vector3f labelPosition = renderData.modelMatrix.col(3).head(3);
  Eigen::Matrix4f modelViewProjection =
      renderData.projectionMatrix * renderData.viewMatrix;
  shaderProgram->setUniform("modelViewProjectionMatrix", modelViewProjection);
  shaderProgram->setUniform("viewMatrix", renderData.viewMatrix);
  shaderProgram->setUniform("labelPosition", labelPosition);
  shaderProgram->setUniform("size",
                            Eigen::Vector2f(2.0f * 0.07f, 0.5f * 0.07f));
  shaderProgram->setUniform("textureSampler", 0);

  texture->bind(gl, GL_TEXTURE0);
  vertexArrayObject.bind();

  glAssert(gl->glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));

  vertexArrayObject.release();
  shaderProgram->release();
}

template <class ElementType>
void Quad::createBuffer(QOpenGLBuffer::Type bufferType, ElementType *data,
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

