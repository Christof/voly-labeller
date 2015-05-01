#include "./labeller.h"
#include <map>
#include <string>
#include <Eigen/LU>
#include "./center_force.h"
#include "../eigen.h"
#include "./anchor_force.h"
/*
#include "./label_collision_force.h"
*/

namespace Forces
{
Labeller::Labeller()
{
  addForce(new CenterForce());
  addForce(new AnchorForce());
  /*
  addForce(new LabelCollisionForce());
  */
}

void Labeller::addLabel(int id, std::string text,
                        Eigen::Vector3f anchorPosition)
{
  labels.push_back(LabelState(id, text, anchorPosition));
}

std::map<int, Eigen::Vector3f>
Labeller::update(const LabellerFrameData &frameData)
{
  std::map<int, Eigen::Vector3f> positions;

  for (auto &force : forces)
    force->beforeAll(labels);

  Eigen::Matrix4f inverseViewProjection = frameData.viewProjection.inverse();

  for (auto &label : labels)
  {
    auto anchor2D = frameData.project(label.anchorPosition);
    label.anchorPosition2D = anchor2D.head<2>();

    auto label2D = frameData.project(label.labelPosition);
    label.labelPosition2D = label2D.head<2>();
    label.labelPositionDepth = label2D.z();

    auto forceOnLabel = Eigen::Vector2f(0, 0);
    for (auto &force : forces)
      forceOnLabel += force->calculate(label, labels, frameData);

    label.labelPosition2D += forceOnLabel * frameData.frameTime;

    Eigen::Vector4f reprojected =
        inverseViewProjection * Eigen::Vector4f(label.labelPosition2D.x(),
                                                label.labelPosition2D.y(),
                                                label.labelPositionDepth, 1);
    reprojected /= reprojected.w();

    label.labelPosition = reprojected.head<3>();

    enforceAnchorDepthForLabel(label, frameData.view);

    positions[label.id] = label.labelPosition;
  }

  return positions;
}

template <class T> void Labeller::addForce(T *force)
{
  forces.push_back(std::unique_ptr<T>(force));
}

void Labeller::enforceAnchorDepthForLabel(LabelState &label,
                                          const Eigen::Matrix4f &viewMatrix)
{
  Eigen::Vector4f anchorView = mul(viewMatrix, label.anchorPosition);
  Eigen::Vector4f labelView = mul(viewMatrix, label.labelPosition);
  labelView.z() = anchorView.z();

  label.labelPosition = (viewMatrix.inverse() * labelView).head<3>();
}

}  // namespace Forces
