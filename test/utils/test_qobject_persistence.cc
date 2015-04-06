#include "../test.h"
#include "../../src/utils/persister.h"

TEST(Test_QObjectPersistence, SaveAndLoadASimpleString)
{
  std::string text = "example text";

  Persister::save(text, "test.xml");

  EXPECT_EQ(text, Persister::load<std::string>("test.xml"));
}
