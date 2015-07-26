#include "./quad.h"
#include <string>
#include <vector>
#include "./gl.h"
#include "./shader_program.h"
#include "./render_object.h"
#include "./object_manager.h"

namespace Graphics
{

int Quad::objectId = -1;

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

void Quad::createBuffers(std::shared_ptr<RenderObject> renderObject,
                         std::shared_ptr<ObjectManager> objectManager)
{
  std::vector<float> positions = { 1.0f, 1.0f,  0.0f, -1.0f, 1.0f,  0.0f,
                                   1.0f, -1.0f, 0.0f, -1.0f, -1.0f, 0.0f };
  renderObject->createBuffer(QOpenGLBuffer::Type::VertexBuffer,
                             positions.data(), "position", 3, 12);
  std::vector<float> texcoords = { 1.0f, 0.0f, 0.0f, 0.0f,
                                   1.0f, 1.0f, 0.0f, 1.0f };
  renderObject->createBuffer(QOpenGLBuffer::Type::VertexBuffer,
                             texcoords.data(), "texcoord", 2, 12);
  std::vector<float> normals = { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
                                 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f };
  std::vector<float> colors = {
    1.0f, 0.0f, 1.0f, 0.5f, 0.0f, 1.0f, 1.0f, 0.5f,
    0.0f, 0.0f, 1.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.5f
  };

  std::vector<uint> indices = { 0, 2, 1, 1, 2, 3 };
  renderObject->createBuffer(QOpenGLBuffer::Type::IndexBuffer, indices.data(),
                             "index", 1, indices.size());

  if (objectId < 0)
    objectId = objectManager->addObject(positions, normals, colors, texcoords,
                                        indices);
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

void Quad::renderToFrameBuffer(Gl *gl, const RenderData &renderData,
                               std::shared_ptr<ObjectManager> objectManager)
{
  if (!renderObject.get())
    initialize(gl, objectManager);

  renderObject->bind();

  setUniforms(renderObject->shaderProgram, renderData);

  draw(gl);

  renderObject->release();
}

}  // namespace Graphics
