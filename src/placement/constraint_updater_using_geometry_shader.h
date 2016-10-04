#ifndef SRC_PLACEMENT_CONSTRAINT_UPDATER_USING_GEOMETRY_SHADER_H_

#define SRC_PLACEMENT_CONSTRAINT_UPDATER_USING_GEOMETRY_SHADER_H_

#include <Eigen/Core>
#include <vector>
#include <memory>
#include "./constraint_updater_base.h"

namespace Graphics
{
  class Gl;
  class ShaderManager;
  class VertexArray;
}

/**
 * \brief
 *
 *
 */
class ConstraintUpdaterUsingGeometryShader : public ConstraintUpdaterBase
{
 public:
  ConstraintUpdaterUsingGeometryShader(
      int width, int height, Graphics::Gl *gl,
      std::shared_ptr<Graphics::ShaderManager> shaderManager);
  virtual ~ConstraintUpdaterUsingGeometryShader();

  void drawConstraintRegionFor(Eigen::Vector2i anchorPosition,
                               Eigen::Vector2i labelSize,
                               Eigen::Vector2i lastAnchorPosition,
                               Eigen::Vector2i lastLabelPosition,
                               Eigen::Vector2i lastLabelSize);
  void drawRegionsForAnchors(std::vector<Eigen::Vector2i> anchorPositions,
                             Eigen::Vector2i labelSize);

  void clear();
  void setIsConnectorShadowEnabled(bool enabled);

 private:
  int width;
  int height;
  Graphics::Gl *gl;
  std::shared_ptr<Graphics::ShaderManager> shaderManager;
  Eigen::Matrix4f pixelToNDC;
  int dialatingShaderId;
  int quadShaderId;

  float labelShadowColor;
  float connectorShadowColor;
  float anchorConstraintColor;

  std::unique_ptr<Graphics::VertexArray> vertexArray;
  std::unique_ptr<Graphics::VertexArray> vertexArrayForConnectors;

  void fillForConnectorShadowRegion(Eigen::Vector2i anchorPosition,
                                 Eigen::Vector2i lastAnchorPosition,
                                 Eigen::Vector2i lastLabelPosition);
};

#endif  // SRC_PLACEMENT_CONSTRAINT_UPDATER_USING_GEOMETRY_SHADER_H_
