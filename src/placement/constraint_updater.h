#ifndef SRC_PLACEMENT_CONSTRAINT_UPDATER_H_

#define SRC_PLACEMENT_CONSTRAINT_UPDATER_H_

#include <Eigen/Core>
#include <vector>
#include "../graphics/gl.h"
#include "../graphics/shader_manager.h"

/**
 * \brief
 *
 *
 */
class ConstraintUpdater
{
 public:
  ConstraintUpdater(Graphics::Gl *gl,
                    std::shared_ptr<Graphics::ShaderManager> shaderManager,
                    int width, int height);

  void addLabel(Eigen::Vector2i anchorPosition, Eigen::Vector2i labelSize,
                Eigen::Vector2i lastAnchorPosition,
                Eigen::Vector2i lastLabelPosition,
                Eigen::Vector2i lastLabelSize);

  void clear();

 private:
  Graphics::Gl *gl;
  std::shared_ptr<Graphics::ShaderManager> shaderManager;
  int width;
  int height;

  int shaderId;
  Eigen::Matrix4f pixelToNDC;

  template <class T>
  void drawPolygon(std::vector<T> polygon);
};

#endif  // SRC_PLACEMENT_CONSTRAINT_UPDATER_H_
