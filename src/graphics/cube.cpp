#include "./cube.h"
#include <vector>
#include <string>
#include "./gl.h"
#include "./shader_manager.h"

namespace Graphics
{

Cube::Cube()
{
}

ObjectData Cube::createBuffers(std::shared_ptr<ObjectManager> objectManager,
                               std::shared_ptr<TextureManager> textureManager,
                               std::shared_ptr<ShaderManager> shaderManager)
{
  createPoints();

  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<float> colors;
  std::vector<float> texCoords;
  for (size_t i = 0; i < points.size(); ++i)
  {
    positions.push_back(points[i].x());
    positions.push_back(points[i].y());
    positions.push_back(points[i].z());

    Eigen::Vector3f normal = points[i].normalized();
    normals.push_back(normal.x());
    normals.push_back(normal.y());
    normals.push_back(normal.z());

    colors.push_back(0);
    colors.push_back(0);
    colors.push_back(1.0f);
    colors.push_back(0.5f);

    texCoords.push_back(0);
    texCoords.push_back(0);
  }

  std::vector<uint> indices = { 0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4,
                                4, 5, 7, 4, 7, 6, 6, 7, 1, 6, 1, 0,
                                0, 2, 4, 0, 4, 6, 1, 7, 5, 1, 5, 3 };

  int shaderProgramId =
      shaderManager->addShader(":/shader/pass.vert", ":/shader/color.frag");
  return objectManager->addObject(positions, normals, colors, texCoords,
                                  indices, shaderProgramId);
}

void Cube::createPoints()
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
}

}  // namespace Graphics
