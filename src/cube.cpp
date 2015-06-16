#include "./cube.h"
#include <vector>
#include <string>
#include "./gl.h"
#include "./shader_program.h"
#include "./render_object.h"

Cube::Cube(std::string vertexShaderPath, std::string fragmentShaderPath)
  : Renderable(vertexShaderPath, fragmentShaderPath)
{
}

void Cube::createBuffers(std::shared_ptr<RenderObject> renderObject)
{
  Eigen::Vector3f frontBottomLeft(-0.5f, -0.5f, 0.5f);
  Eigen::Vector3f frontTopLeft(-0.5f, 0.5f, 0.5f);
  Eigen::Vector3f frontBottomRight(0.5f, -0.5f, 0.5f);
  Eigen::Vector3f frontTopRight(0.5f, 0.5f, 0.5f);

  Eigen::Vector3f backBottomLeft(-0.5f, -0.5f, -0.5f);
  Eigen::Vector3f backTopLeft(-0.5f, 0.5f, -0.5f);
  Eigen::Vector3f backBottomRight(0.5f, -0.5f, -0.5f);
  Eigen::Vector3f backTopRight(0.5f, 0.5f, -0.5f);

  points = std::vector<Eigen::Vector3f>{ frontBottomLeft,  frontTopLeft,
                                         frontBottomRight, frontTopRight,
                                         backBottomRight,  backTopRight,
                                         backBottomLeft,   backTopLeft };

  float *positions = new float[points.size() * 3];
  for (size_t i = 0; i < points.size(); ++i)
  {
    positions[i * 3] = points[i].x();
    positions[i * 3 + 1] = points[i].y();
    positions[i * 3 + 2] = points[i].z();
  }

  int *indices = new int[indexCount]{ 0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4,
                                      4, 5, 7, 4, 7, 6, 6, 7, 1, 6, 1, 0,
                                      0, 2, 4, 0, 4, 6, 1, 7, 5, 1, 5, 3 };

  renderObject->createBuffer(QOpenGLBuffer::Type::VertexBuffer, positions,
                             "position", 3, points.size());
  renderObject->createBuffer(QOpenGLBuffer::Type::IndexBuffer, indices, "index",
                             1, indexCount);

  delete[] indices;
  delete[] positions;
}

void Cube::setUniforms(std::shared_ptr<ShaderProgram> shaderProgram,
                       const RenderData &renderData)
{
  Eigen::Matrix4f modelViewProjection = renderData.projectionMatrix *
                                        renderData.viewMatrix *
                                        renderData.modelMatrix;
  shaderProgram->setUniform("modelViewProjectionMatrix", modelViewProjection);
}

void Cube::draw(Gl *gl)
{
  glAssert(gl->glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0));
}

