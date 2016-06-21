#ifndef SRC_LABELLING_LABEL_POSITIONS_H_

#define SRC_LABELLING_LABEL_POSITIONS_H_

#include <Eigen/Core>
#include <map>

/**
 * \brief Encapsulates positions for labels in 3d and NDC space
 *
 */
class LabelPositions
{
 public:
  LabelPositions() = default;
  LabelPositions(std::map<int, Eigen::Vector3f> positionsNDC,
                 std::map<int, Eigen::Vector3f> positions3d);

  void update(int labelId, Eigen::Vector3f positionNDC,
              Eigen::Vector3f position3d);

  int size();
  int count(int labelId);

  Eigen::Vector3f getNDCFor(int labelId);
  Eigen::Vector3f get3dFor(int labelId);

 private:
  std::map<int, Eigen::Vector3f> positionsNDC;
  std::map<int, Eigen::Vector3f> positions3d;
};

#endif  // SRC_LABELLING_LABEL_POSITIONS_H_
