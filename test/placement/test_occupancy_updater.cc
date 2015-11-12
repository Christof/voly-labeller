#include "../test.h"
#include <Eigen/Core>
#include <thrust/host_vector.h>
#include "../../src/placement/occupancy_updater.h"
#include "../cuda_array_mapper.h"

TEST(Test_OccupancyUpdater, AddLabelInTheMiddle)
{
  auto occupancy = std::make_shared<CudaArrayMapper<float>>(
      4, 4, std::vector<float>(16), cudaCreateChannelDesc<float>());

  OccupancyUpdater occupancyUpdater(occupancy);
  int labelX = 2;
  int labelY = 2;
  int labelWidth = 2;
  int labelHeight = 2;
  occupancyUpdater.addLabel(labelX, labelY, labelWidth, labelHeight);

  auto result = occupancy->copyDataFromGpu();

  ASSERT_EQ(16, result.size());
  EXPECT_EQ(0.0f, result[0]);
  EXPECT_EQ(0.0f, result[1]);
  EXPECT_EQ(0.0f, result[2]);
  EXPECT_EQ(0.0f, result[3]);

  EXPECT_EQ(0.0f, result[4]);
  EXPECT_EQ(1.0f, result[5]);
  EXPECT_EQ(1.0f, result[6]);
  EXPECT_EQ(0.0f, result[7]);

  EXPECT_EQ(0.0f, result[8]);
  EXPECT_EQ(1.0f, result[9]);
  EXPECT_EQ(1.0f, result[10]);
  EXPECT_EQ(0.0f, result[11]);

  EXPECT_EQ(0.0f, result[12]);
  EXPECT_EQ(0.0f, result[13]);
  EXPECT_EQ(0.0f, result[14]);
  EXPECT_EQ(0.0f, result[15]);
}

TEST(Test_OccupancyUpdater, AddLabelInUpperLeftCorner)
{
  auto occupancy = std::make_shared<CudaArrayMapper<float>>(
      4, 4, std::vector<float>(16), cudaCreateChannelDesc<float>());

  OccupancyUpdater occupancyUpdater(occupancy);
  int labelX = 0;
  int labelY = 0;
  int labelWidth = 2;
  int labelHeight = 2;
  occupancyUpdater.addLabel(labelX, labelY, labelWidth, labelHeight);

  auto result = occupancy->copyDataFromGpu();

  ASSERT_EQ(16, result.size());
  EXPECT_EQ(1.0f, result[0]);
  EXPECT_EQ(0.0f, result[1]);
  EXPECT_EQ(0.0f, result[2]);
  EXPECT_EQ(0.0f, result[3]);

  EXPECT_EQ(0.0f, result[4]);
  EXPECT_EQ(0.0f, result[5]);
  EXPECT_EQ(0.0f, result[6]);
  EXPECT_EQ(0.0f, result[7]);

  EXPECT_EQ(0.0f, result[8]);
  EXPECT_EQ(0.0f, result[9]);
  EXPECT_EQ(0.0f, result[10]);
  EXPECT_EQ(0.0f, result[11]);

  EXPECT_EQ(0.0f, result[12]);
  EXPECT_EQ(0.0f, result[13]);
  EXPECT_EQ(0.0f, result[14]);
  EXPECT_EQ(0.0f, result[15]);
}

TEST(Test_OccupancyUpdater, AddLabelInLowerRightCorner)
{
  auto occupancy = std::make_shared<CudaArrayMapper<float>>(
      4, 4, std::vector<float>(16), cudaCreateChannelDesc<float>());

  OccupancyUpdater occupancyUpdater(occupancy);
  int labelX = 3;
  int labelY = 3;
  int labelWidth = 2;
  int labelHeight = 2;
  occupancyUpdater.addLabel(labelX, labelY, labelWidth, labelHeight);

  auto result = occupancy->copyDataFromGpu();

  ASSERT_EQ(16, result.size());
  EXPECT_EQ(0.0f, result[0]);
  EXPECT_EQ(0.0f, result[1]);
  EXPECT_EQ(0.0f, result[2]);
  EXPECT_EQ(0.0f, result[3]);

  EXPECT_EQ(0.0f, result[4]);
  EXPECT_EQ(0.0f, result[5]);
  EXPECT_EQ(0.0f, result[6]);
  EXPECT_EQ(0.0f, result[7]);

  EXPECT_EQ(0.0f, result[8]);
  EXPECT_EQ(0.0f, result[9]);
  EXPECT_EQ(1.0f, result[10]);
  EXPECT_EQ(1.0f, result[11]);

  EXPECT_EQ(0.0f, result[12]);
  EXPECT_EQ(0.0f, result[13]);
  EXPECT_EQ(1.0f, result[14]);
  EXPECT_EQ(1.0f, result[15]);
}

