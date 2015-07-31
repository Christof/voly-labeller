#include "./quad.h"
#include <string>
#include <vector>
#include "./gl.h"
#include "./shader_program.h"
#include "./render_object.h"

namespace Graphics
{

ObjectData Quad::objectData;

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
  std::vector<float> texcoords = { 1.0f, 0.0f, 0.0f, 0.0f,
                                   1.0f, 1.0f, 0.0f, 1.0f };
  std::vector<float> normals = { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
                                 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f };
  std::vector<float> colors = {
    1.0f, 0.0f, 1.0f, 0.5f, 0.0f, 1.0f, 1.0f, 0.5f,
    0.0f, 0.0f, 1.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.5f
  };
  std::vector<uint> indices = { 0, 2, 1, 1, 2, 3 };

  int shaderProgramId =
      objectManager->addShader(":/shader/pass.vert", ":/shader/test.frag");

  if (objectData.vertexSize <= 0)
    objectData = objectManager->addObject(positions, normals, colors, texcoords,
                                          indices, shaderProgramId);
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
  // glAssert(gl->glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
}

void Quad::renderToFrameBuffer(Gl *gl, const RenderData &renderData)
{
  if (!renderObject.get())
    initialize(gl, objectManager);

  setUniforms(renderObject->shaderProgram, renderData);

  objectManager->renderImmediately(objectData);
}

}  // namespace Graphics
