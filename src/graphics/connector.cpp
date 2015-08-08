#include "./connector.h"
#include <vector>
#include "./gl.h"
#include "./shader_program.h"
#include "./render_object.h"

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

void Connector::createBuffers(std::shared_ptr<RenderObject> renderObject,
                              std::shared_ptr<ObjectManager> objectManager)
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
      objectManager->addShader(":/shader/pass.vert", ":/shader/test.frag");
  objectData = objectManager->addObject(positions, normals, colors, texCoords,
                                        indices, shaderProgramId, GL_LINES);
}

void Connector::setUniforms(std::shared_ptr<ShaderProgram> shaderProgram,
                            const RenderData &renderData)
{
  /*
  Eigen::Matrix4f modelViewProjection = renderData.projectionMatrix *
                                        renderData.viewMatrix *
                                        renderData.modelMatrix;
  shaderProgram->setUniform("modelViewProjectionMatrix", modelViewProjection);
  shaderProgram->setUniform("color", color);
  shaderProgram->setUniform("zOffset", zOffset);
  */

  objectData.transform = renderData.modelMatrix;

  objectManager->renderLater(objectData);
}

}  // namespace Graphics

