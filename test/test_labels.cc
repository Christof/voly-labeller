#include <Eigen/Core>
#include "./test.h"
#include "../src/labelling/labels.h"
#include "../src/labelling/label.h"

TEST(Test_Labels, ChangeNotification)
{
  Labels labels;

  int changedLabelId = -1;

  auto removeSubscription =
      labels.subscribe([&changedLabelId](const Label &label)
                       {
                         changedLabelId = label.id;
                       });

  labels.add(Label(1, "Label text", Eigen::Vector3f(1, 2, 3)));

  ASSERT_EQ(1, changedLabelId);

  removeSubscription();
  labels.add(Label(2, "Label text", Eigen::Vector3f(1, 2, 3)));

  ASSERT_EQ(1, changedLabelId);
}
