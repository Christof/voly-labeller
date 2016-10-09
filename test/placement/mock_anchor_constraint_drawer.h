#ifndef TEST_PLACEMENT_MOCK_ANCHOR_CONSTRAINT_DRAWER_H_

#define TEST_PLACEMENT_MOCK_ANCHOR_CONSTRAINT_DRAWER_H_

#include "../../src/placement/anchor_constraint_drawer.h"

class Mock_AnchorConstraintDrawer : public AnchorConstraintDrawer
{
 public:
  std::vector<float> anchors;

  Mock_AnchorConstraintDrawer(int width, int height)
    : AnchorConstraintDrawer(width, height)
  {
  }

  virtual void update(const std::vector<float> &anchors)
  {
    this->anchors = anchors;
  }

  virtual void draw(float color, Eigen::Vector2f halfSize)
  {
  }

  virtual void clear()
  {
  }
};

#endif  // TEST_PLACEMENT_MOCK_ANCHOR_CONSTRAINT_DRAWER_H_
