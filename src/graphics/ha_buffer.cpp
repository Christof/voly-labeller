#include "./ha_buffer.h"
#include <iostream>
#include <algorithm>
#include "./shader_program.h"
#include "./quad.h"

namespace Graphics
{

HABuffer::HABuffer(Eigen::Vector2i size) : size(size)
{
  offsets = new unsigned int[512];
}

HABuffer::~HABuffer()
{
  delete[] offsets;
}

void HABuffer::initialize(Gl *gl)
{
  this->gl = gl;

  quad = std::make_shared<Quad>();
  quad->skipSettingUniforms = true;
  quad->initialize(gl);

  initializeShadersHash();
  initializeBufferHash();

  clearTimer.initialize(gl);
  buildTimer.initialize(gl);
  renderTimer.initialize(gl);
}

void HABuffer::initializeShadersHash()
{
  printf("initShaders %d %d\n", size(0), size(1));

  renderShader = std::make_shared<ShaderProgram>(
      gl, ":shader/renderHABuffer.vert", ":shader/renderHABuffer.frag");
  clearShader = std::make_shared<ShaderProgram>(
      gl, ":shader/clearHABuffer.vert", ":shader/clearHABuffer.frag");
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

void HABuffer::beginAll()
{
  buildTimer.start();

  glAssert(gl->glDisable(GL_CULL_FACE));
  glAssert(gl->glDisable(GL_DEPTH_TEST));
}

void HABuffer::begin(std::shared_ptr<ShaderProgram> shader)
{
  if (lastUsedProgram != shader->getId())
    setUniforms(shader);

  lastUsedProgram = shader->getId();
}

bool HABuffer::endAll()
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

  buildTimer.stop();

  displayStatistics("after render");

  return overflow;
}

void HABuffer::render()
{
  renderTimer.start();

  renderShader->bind();

  renderShader->setUniform("u_ScreenSz", habufferScreenSize);
  renderShader->setUniform("u_HashSz", habufferTableSize);
  renderShader->setUniformAsVec2Array("u_Offsets", offsets, 256);
  renderShader->setUniform("u_Records", RecordsBuffer);
  renderShader->setUniform("u_Counts", CountsBuffer);
  renderShader->setUniform("u_FragmentData", FragmentDataBuffer);

  // Ensure that all global memory write are done before resolving
  glAssert(gl->glMemoryBarrier(GL_SHADER_GLOBAL_ACCESS_BARRIER_BIT_NV));

  glAssert(gl->glDepthMask(GL_FALSE));
  glAssert(gl->glDisable(GL_DEPTH_TEST));

  quad->setShaderProgram(renderShader);
  quad->renderToFrameBuffer(gl, RenderData());

  glAssert(gl->glDepthMask(GL_TRUE));

  renderTimer.stop();

  float tm_threshold = TIMING_THRESHOLD;
  float cleartime = clearTimer.waitResult();
  float buildtime = buildTimer.waitResult();
  float rendertime = renderTimer.waitResult();
  if (cleartime > tm_threshold ||
      buildtime > tm_threshold ||
      rendertime > tm_threshold)
  {
    printf("Clear time %lf ms\n", cleartime);
    printf("Build time %lf ms\n", buildtime);
    printf("Render time %lf ms\n", rendertime);
  }
}

void HABuffer::clear()
{
  clearTimer.start();

  for (int i = 0; i < 512; i++)
  {
    offsets[i] = rand() ^ (rand() << 8) ^ (rand() << 16);
    offsets[i] = offsets[i] % habufferTableSize;
  }

  clearShader->bind();
  clearShader->setUniform("u_NumRecords", habufferNumRecords);
  clearShader->setUniform("u_ScreenSz", habufferScreenSize);
  clearShader->setUniform("u_Records", RecordsBuffer);
  clearShader->setUniform("u_Counts", CountsBuffer);

  // Render the full screen quad

  quad->setShaderProgram(clearShader);
  quad->renderToFrameBuffer(gl, RenderData());

  // Ensure that all global memory write are done before starting to render
  glAssert(gl->glMemoryBarrier(GL_SHADER_GLOBAL_ACCESS_BARRIER_BIT_NV));

  clearTimer.stop();
}

void HABuffer::setUniforms(std::shared_ptr<ShaderProgram> shader)
{
  shader->setUniform("u_NumRecords", habufferNumRecords);
  shader->setUniform("u_ScreenSz", habufferScreenSize);
  shader->setUniform("u_HashSz", habufferTableSize);
  shader->setUniformAsVec2Array("u_Offsets", offsets, 256);

  shader->setUniform("u_ZNear", habufferZNear);
  shader->setUniform("u_ZFar", habufferZFar);
  shader->setUniform("Opacity", habufferOpacity);
  shader->setUniform("u_Records", RecordsBuffer);
  shader->setUniform("u_Counts", CountsBuffer);
  shader->setUniform("u_FragmentData", FragmentDataBuffer);
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
