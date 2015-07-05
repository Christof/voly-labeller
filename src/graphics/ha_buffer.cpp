#include "./ha_buffer.h"
#include <iostream>
#include <algorithm>
#include "./shader_program.h"
#include "./quad.h"

namespace Graphics
{

HABuffer::HABuffer(Eigen::Vector2i size) : size(size)
{
}

void HABuffer::initialize(Gl *gl)
{
  this->gl = gl;

  quad = std::make_shared<Quad>();
  quad->skipSettingUniforms = true;
  quad->initialize(gl);

  initializeShadersHash();
  initializeBufferHash();
}

void HABuffer::initializeShadersHash()
{
  printf("initShaders %d %d\n", size(0), size(1));

  buildShader = std::make_shared<ShaderProgram>(
      gl, ":shader/buildHABuffer.vert", ":shader/buildHABuffer.frag");
  renderShader = std::make_shared<ShaderProgram>(
      gl, ":shader/renderHABuffer.vert", ":shader/renderHABuffer.frag");
  clearShader = std::make_shared<ShaderProgram>(
      gl, ":shader/clearHABuffer.vert", ":shader/clearHABuffer.frag");

  /*
  /// clear shader parameter initialization
  u_NumRecords_clear.init(clearShader, "u_NumRecords");
  u_ScreenSize_clear.init(clearShader, "u_ScreenSz");
  u_Records_clear.init(clearShader, "u_Records");
  u_Counts_clear.init(clearShader, "u_Counts");

  u_Projection_clear.init(clearShader, "u_Projection");

  /// build shader parameter initialization
  u_NumRecords_build.init(buildShader, "u_NumRecords");
  u_ScreenSize_build.init(buildShader, "u_ScreenSz");
  u_HashSize_build.init(buildShader, "u_HashSz");
  u_Offsets_build.init(buildShader, "u_Offsets");
  u_ZNear_build.init(buildShader, "u_ZNear");
  u_ZFar_build.init(buildShader, "u_ZFar");
  u_Opacity_build.init(buildShader, "Opacity");
  u_Records_build.init(buildShader, "u_Records");
  u_Counts_build.init(buildShader, "u_Counts");
  u_FragmentData_build.init(buildShader, "u_FragmentData");

  u_Projection_build.init(buildShader, "u_Projection");
  u_View_build.init(buildShader, "u_View");
#if !USE_INDIRECT
  u_Model_build.init(buildShader, "u_Model");
#endif
#if !USE_TEXTURE
  u_ModelView_IT_build.init(buildShader, "u_ModelView_IT");
#endif

  /// render shader parameter initialization
  // u_NumRecords_render.init(renderShader, "u_NumRecords");
  u_ScreenSize_render.init(renderShader, "u_ScreenSz");
  u_HashSize_render.init(renderShader, "u_HashSz");
  u_Offsets_render.init(renderShader, "u_Offsets");
  u_Records_render.init(renderShader, "u_Records");
  u_Counts_render.init(renderShader, "u_Counts");
  u_FragmentData_render.init(renderShader, "u_FragmentData");

  u_Projection_render.init(renderShader, "u_Projection");
  u_View_render.init(renderShader, "u_View");
  u_Model_render.init(renderShader, "u_Model");

  GL_CHECK_ERROR;
  */
}

void HABuffer::initializeBufferHash()
{
  habufferScreenSize = std::max(size[0], size[1]);
  uint num_records = habufferScreenSize * habufferScreenSize * 8;
  habufferTableSize =
      std::max(habufferScreenSize,
               static_cast<uint>(ceil(sqrt(static_cast<float>(num_records)))));
  habufferNumRecords = habufferTableSize * habufferTableSize;
  habufferCountsSize = habufferScreenSize * habufferScreenSize + 1;
  printf("HA-Buffer: Screen size: %d %d\n"
         "             # records: %d (%d x %d)\n",
         size.x(), size.y(), habufferNumRecords, habufferTableSize,
         habufferTableSize);

  // HA-Buffer records
  if (!RecordsBuffer.isInitialized())
    RecordsBuffer.initialize(gl, habufferNumRecords * sizeof(uint) * 2);
  else
    RecordsBuffer.resize(habufferNumRecords * sizeof(uint) * 2);

  if (!CountsBuffer.isInitialized())
    CountsBuffer.initialize(gl, habufferCountsSize * sizeof(uint));
  else
    CountsBuffer.resize(habufferCountsSize * sizeof(uint));

  if (!FragmentDataBuffer.isInitialized())
    FragmentDataBuffer.initialize(gl,
                                  habufferNumRecords * sizeof(FragmentData));
  else
    FragmentDataBuffer.resize(habufferNumRecords * sizeof(FragmentData));

  // clear counts

  CountsBuffer.clear(0);

  gl->glMemoryBarrier(GL_ALL_BARRIER_BITS);

  printf("[HABuffer] Memory usage: %.2fMB",
         ((habufferNumRecords * sizeof(uint) * 2 +
           habufferNumRecords * sizeof(FragmentData) +
           (habufferScreenSize * habufferScreenSize + 1) * sizeof(uint)) /
          1024) /
             1024.0f);
}

void HABuffer::begin(const RenderData &renderData)
{
  buildShader->bind();
  buildShader->setUniform("u_Projection", renderData.projectionMatrix);
  buildShader->setUniform("u_View", renderData.viewMatrix);

#if !USE_TEXTURE
  Eigen::Matrix4f modelViewMatrixIT = modelViewMatrix.inverse().transpose();
  buildShader->setUniform("u_ModelView_IT", modelViewMatrixIT);
#endif

  // checkGLError("displayRenderHABuffer - before drawModel");

  // Render the model

  glAssert(gl->glDisable(GL_CULL_FACE));
  glAssert(gl->glDisable(GL_DEPTH_TEST));
}

bool HABuffer::end()
{
#if 1
  glAssert(gl->glMemoryBarrier(GL_ALL_BARRIER_BITS));
#endif
  glAssert(gl->glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT));

  uint numInserted = 1;
  CountsBuffer.getData(&numInserted, sizeof(uint),
                       CountsBuffer.getSize() - sizeof(uint));

  bool overflow = false;
  if (numInserted >= habufferNumRecords)
  {
    overflow = true;
    printf("Frame was interrupted: %u\n", numInserted);
  }
  else if (numInserted > habufferNumRecords * 0.8)
  {
    printf("inserted %u / %u\n", numInserted, habufferNumRecords);
  }

  displayStatistics("after render");

  return overflow;
}

void HABuffer::render()
{
  renderShader->bind();
  Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
  renderShader->setUniform("u_Projection", identity);
  renderShader->setUniform("u_View", identity);
  renderShader->setUniform("u_Model", identity);

  // Ensure that all global memory write are done before resolving
  glAssert(gl->glMemoryBarrier(GL_SHADER_GLOBAL_ACCESS_BARRIER_BIT_NV));

  glAssert(gl->glDepthMask(GL_FALSE));
  glAssert(gl->glDisable(GL_DEPTH_TEST));

  quad->setShaderProgram(renderShader);
  quad->render(gl, RenderData());

  glAssert(gl->glDepthMask(GL_TRUE));
  glAssert(gl->glEnable(GL_DEPTH_TEST));

  // TODO(SIRK): print timing
  /*
  float tm_threshold = TIMING_THRESHOLD;
  float cleartime = g_TmClear.waitResult();
  float buildtime = g_TmBuild.waitResult();
  float rendertime = g_TmRender.waitResult();
  if (cleartime > tm_threshold ||
      buildtime > tm_threshold ||
      rendertime > tm_threshold)
  {
    printf("Clear time %lf ms\n", cleartime);
    printf("Build time %lf ms\n", buildtime);
    printf("Render time %lf ms\n", rendertime);
  }
  */
}

void HABuffer::clear()
{
  static uint offsets[512];

  for (int i = 0; i < 512; i++)
  {
    offsets[i] = rand() ^ (rand() << 8) ^ (rand() << 16);
    offsets[i] = offsets[i] % habufferTableSize;
  }

  clearShader->bind();
  Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
  clearShader->setUniform("u_Projection", identity);
  clearShader->setUniform("u_NumRecords", habufferNumRecords);
  clearShader->setUniform("u_ScreenSz", habufferScreenSize);
  clearShader->setUniform("u_Records", RecordsBuffer);
  clearShader->setUniform("u_Counts", CountsBuffer);

  buildShader->setUniform("u_NumRecords", habufferNumRecords);
  buildShader->setUniform("u_ScreenSz", habufferScreenSize);
  buildShader->setUniform("u_HashSz", habufferTableSize);
  buildShader->setUniformAsVec2Array("u_Offsets", offsets, 512);

  buildShader->setUniform("u_ZNear", habufferZNear);
  buildShader->setUniform("u_ZFar", habufferZFar);
  buildShader->setUniform("Opacity", habufferOpacity);
  buildShader->setUniform("u_Records", RecordsBuffer);
  buildShader->setUniform("u_Counts", CountsBuffer);
  buildShader->setUniform("u_FragmentData", FragmentDataBuffer);

  renderShader->setUniform("u_ScreenSz", habufferScreenSize);
  renderShader->setUniform("u_HashSz", habufferTableSize);
  renderShader->setUniformAsVec2Array("u_Offsets", offsets, 512);
  renderShader->setUniform("u_Records", RecordsBuffer);
  renderShader->setUniform("u_Counts", CountsBuffer);
  renderShader->setUniform("u_FragmentData", FragmentDataBuffer);

  // Render the full screen quad
  glAssert(gl->glColorMask(GL_FALSE, GL_FALSE, GL_FALSE,
                           GL_FALSE));  // no effect on screen

  quad->setShaderProgram(clearShader);
  quad->render(gl, RenderData());

  glAssert(gl->glColorMask(GL_TRUE, GL_TRUE, GL_TRUE,
                           GL_TRUE));  // reenable screen drawing

  // Ensure that all global memory write are done before starting to render
  glAssert(gl->glMemoryBarrier(GL_SHADER_GLOBAL_ACCESS_BARRIER_BIT_NV));
}

void HABuffer::displayStatistics(const char *label)
{
  uint *lcounts = new uint[habufferCountsSize];

  CountsBuffer.getData(lcounts, CountsBuffer.getSize());

  int avgdepth = 0;
  int num = 0;
  for (uint c = 0; c < habufferCountsSize - 1; c++)
  {
    if (lcounts[c] > 0)
    {
      num++;
      avgdepth += lcounts[c];
    }
  }
  if (num == 0)
    num = 1;

  double rec_percentage = lcounts[habufferCountsSize - 1] /
                          static_cast<double>(habufferNumRecords) * 100.0;

  if (rec_percentage > 80.0)
  {
    std::cerr << label << " habufferCountsSize:" << habufferCountsSize
              << "<avg:" << avgdepth / static_cast<float>(num)
              << " max: " << lcounts[habufferCountsSize - 1] << "/"
              << habufferNumRecords << "(" << rec_percentage << "% " << '>'
              << std::endl;
  }

  delete[] lcounts;
}

}  // namespace Graphics
