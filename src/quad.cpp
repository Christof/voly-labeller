#include "./quad.h"
#include <Eigen/LU>
#include <iostream>
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

  Eigen::Vector3f cameraPosition = renderData.cameraPosition;
  Eigen::Vector3f cameraUp =
      renderData.viewMatrix.block<3, 3>(0, 0) * Eigen::Vector3f(0, 1, 0);
  Eigen::Vector3f labelPosition = renderData.modelMatrix.col(3).head(3);
  Eigen::Vector3f look = (cameraPosition - labelPosition).normalized();
  Eigen::Vector3f right = cameraUp.cross(look);
  Eigen::Vector3f up = look.cross(right);

  Eigen::Matrix4f modelMatrix = Eigen::Matrix4f::Identity();
  modelMatrix.col(0).head(3) = right;
  modelMatrix.col(1).head(3) = up;
  modelMatrix.col(2).head(3) = look;
  modelMatrix.col(3).head(3) = labelPosition;
  /*
  Eigen::Matrix3f inverseRot;
  Eigen::Matrix3f rot = renderData.viewMatrix.block<3, 3>(0, 0);
  bool invertible;
  rot.computeInverseWithCheck(inverseRot, invertible);
  Eigen::Matrix4f inverseView = Eigen::Matrix4f::Identity();
  inverseView.block<3, 3>(0, 0) = inverseRot;

  std::cout << "view: " << renderData.viewMatrix.format(Eigen::IOFormat()) <<
  std::endl;
  std::cout << "inverse: " << inverseView.format(Eigen::IOFormat()) <<
  std::endl;

  Eigen::Matrix4f modelView =
      renderData.viewMatrix * inverseView * renderData.modelMatrix;

  modelView(0, 1) = modelView(0, 2) = 0.0f;
  modelView(1, 0) = modelView(1, 2) = 0.0f;
  modelView(2, 0) = modelView(2, 1) = 0.0f;
  */

  Eigen::Matrix4f modelViewProjection =
      renderData.projectionMatrix * renderData.viewMatrix * modelMatrix;
  shaderProgram->setUniform("modelViewProjectionMatrix", modelViewProjection);
  shaderProgram->setUniform("modelMatrix", renderData.modelMatrix);
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

