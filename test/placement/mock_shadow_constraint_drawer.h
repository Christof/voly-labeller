#ifndef TEST_PLACEMENT_MOCK_SHADOW_CONSTRAINT_DRAWER_H_

#define TEST_PLACEMENT_MOCK_SHADOW_CONSTRAINT_DRAWER_H_

#include "../../src/placement/shadow_constraint_drawer.h"

class Mock_ShadowConstraintDrawer : public ShadowConstraintDrawer
{
 public:
  std::vector<float> sources;
  std::vector<float> starts;
  std::vector<float> ends;

  Mock_ShadowConstraintDrawer(int width, int height)
    : ShadowConstraintDrawer(width, height)
  {
  }

  virtual void update(const std::vector<float> &sources,
                      const std::vector<float> &starts,
                      const std::vector<float> &ends)
  {
    this->sources = sources;
    this->starts = starts;
    this->ends = ends;
  }

  virtual void draw(float color, Eigen::Vector2f halfSize)
  {
  }

  virtual void clear()
  {
  }
};

#endif  // TEST_PLACEMENT_MOCK_SHADOW_CONSTRAINT_DRAWER_H_
