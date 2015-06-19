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
      "}\n\n\n"
      "void main() {}\n", output.toStdString());
}

TEST(Test_ShaderProgram, commentedIncludesAreIgnored)
{
  auto output = ShaderProgram::readFileAndHandleIncludes(
      absolutePathOfProjectRelativePath(QString("test/test_shader/commented_include.vert")));

  ASSERT_EQ(
      "#version 330\n"
      "\n"
      "// #include \"test_common_shader_functions.glsl\"\n"
      "\n"
      "void main() {}\n", output.toStdString());
}

TEST(Test_ShaderProgram, removesVersionFromIncludedFile)
{
  auto output = ShaderProgram::readFileAndHandleIncludes(
      absolutePathOfProjectRelativePath(QString("test/test_shader/multiple_version_statements.vert")));

  ASSERT_EQ(
      "#version 330\n"
      "\n"
      "\n"
      "\n"
      "void something() {}\n"
      "\n"
      "\n"
      "void main() {}\n", output.toStdString());
}

TEST(Test_ShaderProgram, multipleIncludes)
{
  auto output = ShaderProgram::readFileAndHandleIncludes(
      absolutePathOfProjectRelativePath(QString("test/test_shader/multiple_includes.vert")));

  ASSERT_EQ(
      "#version 330\n"
      "\n"
      "\n"
      "\n"
      "void something() {}\n"
      "\n"
      "float special()\n"
      "{\n"
      "  return 47.11f;\n"
      "}\n\n\n"
      "void main() {}\n", output.toStdString());
}

