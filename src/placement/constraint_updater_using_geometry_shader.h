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
class ConstraintDrawer;

/**
 * \brief ConstraintUpdaterBase implementation using a geometry shader
 *
 * The dilation in the geometry shader `constraint.geom` is adapted from
 * Hasselgren, J., Akenine-Möller, T., & Ohlsson, L. (2005).
 * Conservative rasterization. GPU Gems, 2, 677–690. article.
 * (http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter42.html)
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
  void finish();
  void setIsConnectorShadowEnabled(bool enabled);

 private:
  int width;
  int height;
  Graphics::Gl *gl;
  std::shared_ptr<Graphics::ShaderManager> shaderManager;
  Eigen::Matrix4f pixelToNDC;
  std::unique_ptr<ConstraintDrawer> quadDrawer;
  std::unique_ptr<ConstraintDrawer> dilatingDrawer;

  float labelShadowColor;
  float connectorShadowColor;
  float anchorConstraintColor;

  std::unique_ptr<Graphics::VertexArray> vertexArray;
  std::unique_ptr<Graphics::VertexArray> vertexArrayForConnectors;
  std::unique_ptr<Graphics::VertexArray> vertexArrayForAnchors;

  std::vector<float> sources;
  std::vector<float> starts;
  std::vector<float> ends;

  std::vector<float> anchors;
  std::vector<float> connectorStart;
  std::vector<float> connectorEnd;

  Eigen::Vector2f labelSize;

  void addConnectorShadow(Eigen::Vector2i anchor, Eigen::Vector2i start,
                          Eigen::Vector2i end);
  void addLineShadow(Eigen::Vector2f anchor, Eigen::Vector2f start,
                     Eigen::Vector2f end);
  std::vector<Eigen::Vector2f> getCornersFor(Eigen::Vector2i position,
                                             Eigen::Vector2f halfSize);
};

#endif  // SRC_PLACEMENT_CONSTRAINT_UPDATER_USING_GEOMETRY_SHADER_H_
