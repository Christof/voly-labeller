#include "./connector.h"
#include <vector>
#include "./gl.h"
#include "./shader_program.h"
#include "./shader_manager.h"

namespace Graphics
{

Connector::Connector(Eigen::Vector3f anchor, Eigen::Vector3f label)
  : Connector(std::vector<Eigen::Vector3f>{ anchor, label })
{
}

Connector::Connector(std::vector<Eigen::Vector3f> points)
  : color(1, 0, 0, 1), points(points)
{
}

Connector::~Connector()
{
}

ObjectData
Connector::createBuffers(std::shared_ptr<ObjectManager> objectManager,
                         std::shared_ptr<TextureManager> textureManager,
                         std::shared_ptr<ShaderManager> shaderManager)
{
  std::vector<float> positions;
  std::vector<float> normals(points.size() * 3, 0.0f);
  std::vector<float> colors;
  std::vector<float> texCoords(points.size() * 2, 0.0f);
  std::vector<uint> indices;
  for (size_t i = 0; i < points.size(); ++i)
  {
    positions.push_back(points[i].x());
    positions.push_back(points[i].y());
    positions.push_back(points[i].z());

    colors.push_back(color.x());
    colors.push_back(color.y());
    colors.push_back(color.z());
    colors.push_back(color.w());

    indices.push_back(i);
  }

  int shaderProgramId =
      shaderManager->addShader(":/shader/pass.vert", ":/shader/test.frag");
  return objectManager->addObject(positions, normals, colors, texCoords,
                                  indices, shaderProgramId, GL_LINES);
}

}  // namespace Graphics

