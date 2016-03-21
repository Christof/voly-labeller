#include <Eigen/Core>
#include "./test.h"
#include "../src/labelling/labels.h"
#include "../src/labelling/label.h"

TEST(Test_Labels, ChangeNotification)
{
  Labels labels;

  int changedLabelId = -1;
  Labels::Action receivedAction;

  auto removeSubscription = labels.subscribe([&changedLabelId, &receivedAction](
      Labels::Action action, const Label &label)
                                             {
                                               receivedAction = action;
                                               changedLabelId = label.id;
                                             });

  labels.add(Label(1, "Label text", Eigen::Vector3f(1, 2, 3)));

  ASSERT_EQ(1, changedLabelId);
  ASSERT_EQ(Labels::Action::Add, receivedAction);

  removeSubscription();
  labels.add(Label(2, "Label text", Eigen::Vector3f(1, 2, 3)));

  ASSERT_EQ(1, changedLabelId);
}

TEST(Test_Labels, Remove)
{
  Labels labels;

  Label label(1, "Label text", Eigen::Vector3f(1, 2, 3));
  labels.add(label);

  ASSERT_EQ(1, labels.count());

  int changedLabelId = -1;
  Labels::Action receivedAction;
  auto removeSubscription = labels.subscribe([&changedLabelId, &receivedAction](
      Labels::Action action, const Label &label)
                                             {
                                               receivedAction = action;
                                               changedLabelId = label.id;
                                             });
  labels.remove(label);

  EXPECT_EQ(0, labels.count());
  EXPECT_EQ(1, changedLabelId);
  EXPECT_EQ(Labels::Action::Delete, receivedAction);

  removeSubscription();
}

TEST(Test_Labels, UpdateTriggersANotification)
{
  Labels labels;

  Label label(1, "Label text", Eigen::Vector3f(1, 2, 3));
  labels.add(label);

  label.text = "changed text";

  int changedLabelId = -1;
  Labels::Action receivedAction;
  auto removeSubscription = labels.subscribe([&changedLabelId, &receivedAction](
      Labels::Action action, const Label &label)
                                             {
                                               receivedAction = action;
                                               changedLabelId = label.id;
                                             });

  labels.update(label);

  EXPECT_EQ(1, changedLabelId);
  EXPECT_EQ(Labels::Action::Update, receivedAction);
  EXPECT_EQ("changed text", labels.getById(1).text);

  removeSubscription();
}

TEST(Test_Labels, GetLabelsReturnsCopyOfAllLabels)
{
  Labels labels;

  Label label(1, "Label text", Eigen::Vector3f(1, 2, 3));
  labels.add(label);
  Label label2(2, "Other Label text", Eigen::Vector3f(1, 2, 3));
  labels.add(label2);

  auto labelsVector = labels.getLabels();
  EXPECT_EQ(2, labels.count());
  EXPECT_EQ(label, labelsVector[0]);
  EXPECT_EQ(label2, labelsVector[1]);

  labelsVector[0].text = "Changed text";

  EXPECT_EQ(label.text, labels.getLabels()[0].text);
}

TEST(Test_Labels, ClearRemovesAllLabels)
{
  Labels labels;

  Label label(1, "Label text", Eigen::Vector3f(1, 2, 3));
  labels.add(label);
  Label label2(2, "Other Label text", Eigen::Vector3f(1, 2, 3));
  labels.add(label2);

  labels.clear();

  EXPECT_EQ(0, labels.count());
}

TEST(Test_Labels, GetByIdReturnsEmptyLabelForNotExistingId)
{
  Labels labels;

  auto label = labels.getById(1);

  EXPECT_EQ(-1, label.id);
  EXPECT_EQ("", label.text);
}

