#include "../test.h"
#include "../../src/graphics/buffer_hole_manager.h"

TEST(Test_BufferHoleManager, reserveOnce)
{
  Graphics::BufferHoleManager manager(10);

  uint offset = 0;
  bool success = manager.reserve(5, offset);

  EXPECT_TRUE(success);
  EXPECT_EQ(0, offset);
}

TEST(Test_BufferHoleManager, reserveTwice)
{
  Graphics::BufferHoleManager manager(10);

  uint offset = 0;
  manager.reserve(5, offset);
  bool success = manager.reserve(3, offset);

  EXPECT_TRUE(success);
  EXPECT_EQ(5, offset);
}

TEST(Test_BufferHoleManager, reserveForChunkLargerAsBufferReturnsFalse)
{
  Graphics::BufferHoleManager manager(10);

  uint offset = 0;
  bool success = manager.reserve(11, offset);

  EXPECT_FALSE(success);
}

TEST(Test_BufferHoleManager, reserveIfNohtingIsFreeAnyMoreReturnsFalse)
{
  Graphics::BufferHoleManager manager(10);

  uint offset = 0;
  EXPECT_TRUE(manager.reserve(5, offset));
  EXPECT_TRUE(manager.reserve(5, offset));

  bool success = manager.reserve(3, offset);

  EXPECT_FALSE(success);
}
