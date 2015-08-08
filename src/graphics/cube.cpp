#include "./cube.h"
#include <vector>
#include <string>
#include "./gl.h"
#include "./shader_program.h"
#include "./render_object.h"

namespace Graphics
{

Cube::Cube()
{
}

void Cube::createBuffers(std::shared_ptr<RenderObject> renderObject,
                         std::shared_ptr<ObjectManager> objectManager)
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
    colors.push_back(0);
    colors.push_back(1);

    texCoords.push_back(0);
    texCoords.push_back(0);
  }

  std::vector<uint> indices = { 0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4,
                                4, 5, 7, 4, 7, 6, 6, 7, 1, 6, 1, 0,
                                0, 2, 4, 0, 4, 6, 1, 7, 5, 1, 5, 3 };

  int shaderProgramId =
      objectManager->addShader(":/shader/pass.vert", ":/shader/test.frag");
  objectData = objectManager->addObject(positions, normals, colors, texCoords,
                                        indices, shaderProgramId);
}

void Cube::setUniforms(std::shared_ptr<ShaderProgram> shaderProgram,
                       const RenderData &renderData)
{
  objectData.transform = renderData.modelMatrix;
  objectManager->renderLater(objectData);
  /*
  Eigen::Matrix4f modelViewProjection =
      renderData.projectionMatrix * renderData.viewMatrix *
      renderData.modelMatrix;
  shaderProgram->setUniform("modelViewProjectionMatrix", modelViewProjection);
  */
}

}  // namespace Graphics
