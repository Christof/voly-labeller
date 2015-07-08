#include "./quad.h"
#include <string>
#include "./gl.h"
#include "./shader_program.h"
#include "./render_object.h"

namespace Graphics
{


Quad::Quad(std::string vertexShaderFilename, std::string fragmentShaderFilename)
  : Renderable(vertexShaderFilename, fragmentShaderFilename)
{
}

Quad::Quad() : Renderable(":shader/label.vert", ":shader/label.frag")
{
}

Quad::~Quad()
{
}

void Quad::createBuffers(std::shared_ptr<RenderObject> renderObject)
{
  float positions[12]{ 1.0f, 1.0f,  0.0f, -1.0f, 1.0f,  0.0f,
                       1.0f, -1.0f, 0.0f, -1.0f, -1.0f, 0.0f };
  renderObject->createBuffer(QOpenGLBuffer::Type::VertexBuffer, positions,
                             "position", 3, 12);
  float texcoords[8]{ 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f };
  renderObject->createBuffer(QOpenGLBuffer::Type::VertexBuffer, texcoords,
                             "texcoord", 2, 12);
}

void Quad::setUniforms(std::shared_ptr<ShaderProgram> shader,
                       const RenderData &renderData)
{
  if (skipSettingUniforms)
    return;

  Eigen::Matrix4f modelViewProjection =
      renderData.projectionMatrix * renderData.viewMatrix;
  shader->setUniform("modelViewProjectionMatrix", modelViewProjection);
  shader->setUniform("viewMatrix", renderData.viewMatrix);
  shader->setUniform("modelMatrix", renderData.modelMatrix);
  shader->setUniform("textureSampler", 0);
}

void Quad::draw(Gl *gl)
{
  glAssert(gl->glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
}

}  // namespace Graphics
