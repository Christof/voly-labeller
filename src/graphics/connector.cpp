#include "./connector.h"
#include <vector>
#include "./gl.h"
#include "./shader_program.h"
#include "./render_object.h"
#include "./object_manager.h"

namespace Graphics
{

Connector::Connector(Eigen::Vector3f anchor, Eigen::Vector3f label)
  : Connector(std::vector<Eigen::Vector3f>{ anchor, label })
{
}

Connector::Connector(std::vector<Eigen::Vector3f> points)
  : Renderable(":shader/line.vert", ":shader/line.frag"), points(points)
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

    colors.push_back(1);
    colors.push_back(0);
    colors.push_back(0);
    colors.push_back(1);

    indices.push_back(i);
  }

  int shaderProgramId =
      objectManager->addShader(":/shader/pass.vert", ":/shader/test.frag");
  objectId = objectManager->addObject(positions, normals, colors, texCoords,
                                      indices, shaderProgramId, GL_LINES);
}

void Connector::setUniforms(std::shared_ptr<ShaderProgram> shaderProgram,
                            const RenderData &renderData)
{
  Eigen::Matrix4f modelViewProjection = renderData.projectionMatrix *
                                        renderData.viewMatrix *
                                        renderData.modelMatrix;
  shaderProgram->setUniform("modelViewProjectionMatrix", modelViewProjection);
  shaderProgram->setUniform("color", color);
  shaderProgram->setUniform("zOffset", zOffset);

  objectManager->renderLater(objectId, renderData.modelMatrix);
}

void Connector::draw(Gl *gl)
{
  // gl->glLineWidth(lineWidth);
  // glAssert(gl->glDrawArrays(GL_LINES, 0, points.size()));
}

}  // namespace Graphics

