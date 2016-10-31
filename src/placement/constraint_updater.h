#ifndef SRC_PLACEMENT_CONSTRAINT_UPDATER_H_

#define SRC_PLACEMENT_CONSTRAINT_UPDATER_H_

#include <Eigen/Core>
#include <vector>
#include <string>
#include <memory>

class ShadowConstraintDrawer;
class AnchorConstraintDrawer;

/**
 * \brief ConstraintUpdaterBase implementation using a geometry shader
 *
 * The dilation in the geometry shader `constraint.geom` is adapted from
 * Hasselgren, J., Akenine-Möller, T., & Ohlsson, L. (2005).
 * Conservative rasterization. GPU Gems, 2, 677–690. article.
 * (http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter42.html)
 */
class ConstraintUpdater
{
 public:
  ConstraintUpdater(
      int bufferWidth, int bufferHeight,
      std::shared_ptr<AnchorConstraintDrawer> anchorConstraintDrawer,
      std::shared_ptr<ShadowConstraintDrawer> connectorShadowDrawer,
      std::shared_ptr<ShadowConstraintDrawer> shadowConstraintDrawer,
      float scaleFactor);
  virtual ~ConstraintUpdater();

  void drawConstraintRegionFor(Eigen::Vector2i anchorPosition,
                               Eigen::Vector2i labelSize,
                               Eigen::Vector2i lastAnchorPosition,
                               Eigen::Vector2i lastLabelPosition,
                               Eigen::Vector2i lastLabelSize);
  void drawRegionsForAnchors(std::vector<Eigen::Vector2f> anchorPositions,
                             Eigen::Vector2i labelSize);

  void clear();
  void finish();
  void setIsConnectorShadowEnabled(bool enabled);
  void save(std::string filename);

 private:
  int width;
  int height;
  std::shared_ptr<AnchorConstraintDrawer> anchorConstraintDrawer;
  std::shared_ptr<ShadowConstraintDrawer> connectorShadowDrawer;
  std::shared_ptr<ShadowConstraintDrawer> shadowConstraintDrawer;
  float scaleFactor;

  float labelShadowColor;
  float connectorShadowColor;
  float anchorConstraintColor;

  std::vector<float> sources;
  std::vector<float> starts;
  std::vector<float> ends;

  std::vector<float> anchors;
  std::vector<float> connectorStart;
  std::vector<float> connectorEnd;

  Eigen::Vector2f labelSize;
  Eigen::Vector2f borderPixel;

  bool isConnectorShadowEnabled = true;

  void addConnectorShadow(Eigen::Vector2i anchor, Eigen::Vector2i start,
                          Eigen::Vector2i end);
  void addLabelShadow(Eigen::Vector2f anchor, Eigen::Vector2i lastLabelPosition,
                      Eigen::Vector2f lastHalfSize);
  void addLineShadow(Eigen::Vector2f anchor, Eigen::Vector2f start,
                     Eigen::Vector2f end);
  std::vector<Eigen::Vector2f> getCornersFor(Eigen::Vector2i position,
                                             Eigen::Vector2f halfSize);
};

#endif  // SRC_PLACEMENT_CONSTRAINT_UPDATER_H_
