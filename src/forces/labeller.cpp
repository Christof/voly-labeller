#include "./labeller.h"
#include <map>
#include <string>
#include <Eigen/LU>
#include "./center_force.h"
#include "../eigen_qdebug.h"
/*
#include "./anchor_force.h"
#include "./label_collision_force.h"
#include "./anchor_depth_force.h"
*/

namespace Forces
{
Labeller::Labeller()
{
  addForce(new CenterForce());
  /*
  addForce(new AnchorForce());
  addForce(new LabelCollisionForce());
  addForce(new AnchorDepthForce());
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

  qDebug() << "viewProjection "<< frameData.viewProjection;
  Eigen::Matrix4f inverseViewProjection = frameData.viewProjection.inverse();
  qDebug() << "inverse" << inverseViewProjection;

  for (auto &label : labels)
  {
    auto anchor2D = frameData.project(label.anchorPosition);
    label.anchorPosition2D = anchor2D.head<2>();

    auto label2D = frameData.project(label.labelPosition);
    label.labelPosition2D = label2D.head<2>();
    label.labelPositionDepth = label2D.z();

    qDebug() << "3D: " << label.labelPosition;
    qDebug() << "2D: " << label.labelPosition2D;
    qDebug() << "depth: " << label.labelPositionDepth;

    auto forceOnLabel = Eigen::Vector2f(0, 0);
    for (auto &force : forces)
      forceOnLabel += force->calculate(label, labels, frameData);

    label.labelPosition2D += forceOnLabel * frameData.frameTime;

    Eigen::Vector4f reprojected =
        inverseViewProjection * Eigen::Vector4f(label.labelPosition2D.x(),
                                                label.labelPosition2D.y(),
                                                label.labelPositionDepth, 1);
    reprojected /= reprojected.w();

    qDebug() << "3Dn: " << reprojected;
    label.labelPosition = reprojected.head<3>();


    positions[label.id] = label.labelPosition;
  }

  return positions;
}

template <class T> void Labeller::addForce(T *force)
{
  forces.push_back(std::unique_ptr<T>(force));
}
}  // namespace Forces
