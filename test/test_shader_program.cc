#include "./test.h"
#include "../src/shader_program.h"
#include "../src/utils/path_helper.h"

TEST(Test_ShaderProgram, replacesASingleInclude)
{
  auto output = ShaderProgram::readFileAndHandleIncludes(
      absolutePathOfProjectRelativePath(QString("test/test_shader/single_include.vert")));

  ASSERT_EQ(
      "#version 330\n"
      "\n"
      "float special()\n"
      "{\n"
      "  return 47.11f;\n"
      "}\n\n", output.toStdString());
}
