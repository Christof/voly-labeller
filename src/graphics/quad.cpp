#include "./quad.h"
#include <string>
#include <vector>
#include "./gl.h"
#include "./shader_program.h"
#include "./shader_manager.h"

namespace Graphics
{

ObjectData Quad::staticObjectData;

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

ObjectData Quad::createBuffers(std::shared_ptr<ObjectManager> objectManager,
                               std::shared_ptr<TextureManager> textureManager,
                               std::shared_ptr<ShaderManager> shaderManager)
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
      shaderManager->addShader(vertexShaderFilename, fragmentShaderFilename);

  if (!staticObjectData.isInitialized())
    staticObjectData = objectManager->addObject(
        positions, normals, colors, texcoords, indices, shaderProgramId);

  return objectManager->cloneForDifferentShader(staticObjectData,
                                                shaderProgramId);
}

}  // namespace Graphics
