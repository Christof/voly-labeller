#include "./connector.h"
#include "./gl.h"
#include "./shader_program.h"
#include "./render_object.h"

Connector::Connector(Eigen::Vector3f anchor, Eigen::Vector3f label)
  : Connector(std::vector<Eigen::Vector3f> { anchor, label })
{
}

Connector::Connector(std::vector<Eigen::Vector3f> points)
  : Renderable(":shader/line.vert", ":shader/line.frag"), points(points)
{
}

Connector::~Connector()
{
}

void Connector::createBuffers(std::shared_ptr<RenderObject> renderObject)
{
  float *positions = new float[3 * points.size()];
  for (size_t i = 0; i < points.size(); ++i)
  {
    positions[i * 3] = points[i].x();
    positions[i * 3 + 1] = points[i].y();
    positions[i * 3 + 2] = points[i].z();
  }

  renderObject->createBuffer(QOpenGLBuffer::Type::VertexBuffer, positions,
                             "position", 3, points.size());

  delete[] positions;
}

void Connector::setUniforms(std::shared_ptr<ShaderProgram> shaderProgram,
                            const RenderData &renderData)
{
  Eigen::Matrix4f modelViewProjection =
      renderData.projectionMatrix * renderData.viewMatrix;
  shaderProgram->setUniform("modelViewProjectionMatrix", modelViewProjection);
  shaderProgram->setUniform("color", color);
}

void Connector::draw(Gl *gl)
{
  gl->glLineWidth(3.0f);
  glAssert(gl->glDrawArrays(GL_LINES, 0, points.size()));
}

