#include "./connector.h"
#include "./gl.h"
#include "./shader_program.h"
#include "./render_object.h"

Connector::Connector(Eigen::Vector3f anchor, Eigen::Vector3f label)
  : Renderable(":shader/line.vert", ":shader/line.frag"), anchor(anchor),
    label(label)
{
}

Connector::~Connector()
{
}

void Connector::createBuffers(std::shared_ptr<RenderObject> renderObject)
{
  float positions[6]{ anchor.x(), anchor.y(), anchor.z(),
                      label.x(),  label.y(),  label.z() };
  renderObject->createBuffer(QOpenGLBuffer::Type::VertexBuffer, positions,
                             "position", 3, 6);
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
  glAssert(gl->glDrawArrays(GL_LINES, 0, 2));
}

