#include "./test.h"
#include "../src/shader_program.h"
#include "../src/utils/path_helper.h"

TEST(Test_ShaderProgram, readFileAndHandleIncludes)
{
  auto output = ShaderProgram::readFileAndHandleIncludes(
      absolutePathOfProjectRelativePath(QString("test/test_shader.vert")));

  ASSERT_EQ("#version 330\n", output);
}
