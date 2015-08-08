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
  : vertexShaderFilename(vertexShaderFilename),
    fragmentShaderFilename(fragmentShaderFilename)
{
}

Quad::Quad() : Quad(":/shader/pass.vert", ":/shader/test.frag")
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
      objectManager->addShader(vertexShaderFilename, fragmentShaderFilename);

  if (!objectData.isInitialized())
    objectData = objectManager->addObject(positions, normals, colors, texcoords,
                                          indices, shaderProgramId);
}

void Quad::setUniforms(std::shared_ptr<ShaderProgram> shader,
                       const RenderData &renderData)
{
  objectData.transform = renderData.modelMatrix;
  objectManager->renderLater(objectData);
}

}  // namespace Graphics
