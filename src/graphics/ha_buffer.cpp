#include "./ha_buffer.h"
#include <QLoggingCategory>
#include <algorithm>
#include "./shader_program.h"
#include "./screen_quad.h"
#include "./object_manager.h"

namespace Graphics
{

QLoggingCategory channel("Graphics.HABuffer");

HABuffer::HABuffer(Eigen::Vector2i size) : size(size)
{
  offsets = new unsigned int[512];
}

HABuffer::~HABuffer()
{
  delete[] offsets;
}

void HABuffer::initialize(Gl *gl, std::shared_ptr<ObjectManager> objectManager,
                          std::shared_ptr<TextureManager> textureManager,
                          std::shared_ptr<ShaderManager> shaderManager)
{
  this->gl = gl;
  this->objectManager = objectManager;

  clearQuad = std::make_shared<ScreenQuad>(":shader/clearHABuffer.vert",
                                           ":shader/clearHABuffer.frag");
  clearQuad->skipSettingUniforms = true;
  clearQuad->initialize(gl, objectManager, textureManager, shaderManager);

  renderQuad = std::make_shared<ScreenQuad>(":shader/renderHABuffer.vert",
                                            ":shader/renderHABuffer.frag");
  renderQuad->skipSettingUniforms = true;
  renderQuad->initialize(gl, objectManager, textureManager, shaderManager);

  initializeShadersHash();
  initializeBufferHash();

  clearTimer.initialize(gl);
  buildTimer.initialize(gl);
  renderTimer.initialize(gl);
}

void HABuffer::updateNearAndFarPlanes(float near, float far)
{
  zNear = near;
  zFar = far;
}

void HABuffer::initializeShadersHash()
{
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
  qCDebug(channel, "Screen size: %d %d\n# records: %d (%d x %d)\n", size.x(),
          size.y(), habufferNumRecords, habufferTableSize, habufferTableSize);

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
    FragmentDataBuffer.initialize(gl, habufferNumRecords * FRAGMENT_DATA_SIZE);
  else
    FragmentDataBuffer.resize(habufferNumRecords * FRAGMENT_DATA_SIZE);

  // clear counts
  CountsBuffer.clear(0);

  gl->glMemoryBarrier(GL_ALL_BARRIER_BITS);

  qCDebug(channel) << "Memory usage:"
                   << ((habufferNumRecords * sizeof(uint) * 2 +
                        habufferNumRecords * FRAGMENT_DATA_SIZE +
                        (habufferScreenSize * habufferScreenSize + 1) *
                            sizeof(uint)) /
                       1024) /
                          1024.0f << "MB";
}

void HABuffer::clearAndPrepare()
{
  clearTimer.start();

  for (int i = 0; i < 512; i++)
  {
    offsets[i] = rand() ^ (rand() << 8) ^ (rand() << 16);
    offsets[i] = offsets[i] % habufferTableSize;
  }

  auto clearShader = clearQuad->getShaderProgram();
  clearShader->bind();
  clearShader->setUniform("u_NumRecords", habufferNumRecords);
  clearShader->setUniform("u_ScreenSz", habufferScreenSize);
  clearShader->setUniform("u_Records", RecordsBuffer);
  clearShader->setUniform("u_Counts", CountsBuffer);

  clearQuad->render(gl, objectManager, RenderData());

  if (wireframe)
    gl->glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  // Ensure that all global memory write are done before starting to render
  glAssert(gl->glMemoryBarrier(GL_SHADER_GLOBAL_ACCESS_BARRIER_BIT_NV));

  clearTimer.stop();
  buildTimer.start();

  glAssert(gl->glDisable(GL_CULL_FACE));
  glAssert(gl->glDisable(GL_DEPTH_TEST));
}

void HABuffer::begin(std::shared_ptr<ShaderProgram> shader)
{
  // TODO(sirk): re-enable optimization after change to ObjectManager
  // if (lastUsedProgram != shader->getId())
  setUniforms(shader);

  lastUsedProgram = shader->getId();
}

void HABuffer::render(const RenderData &renderData)
{
  if (wireframe)
    gl->glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  syncAndGetCounts();

  renderTimer.start();

  auto renderShader = renderQuad->getShaderProgram();
  renderShader->bind();

  renderShader->setUniform("u_ScreenSz", habufferScreenSize);
  renderShader->setUniform("u_HashSz", habufferTableSize);
  renderShader->setUniformAsVec2Array("u_Offsets", offsets, 256);
  renderShader->setUniform("u_Records", RecordsBuffer);
  renderShader->setUniform("u_Counts", CountsBuffer);
  renderShader->setUniform("u_FragmentData", FragmentDataBuffer);
  Eigen::Matrix4f viewProjectionMatrix =
      renderData.projectionMatrix * renderData.viewMatrix;
  renderShader->setUniform("viewProjectionMatrix", viewProjectionMatrix);
  renderShader->setUniform("viewMatrix", renderData.viewMatrix);

  // Ensure that all global memory write are done before resolving
  glAssert(gl->glMemoryBarrier(GL_SHADER_GLOBAL_ACCESS_BARRIER_BIT_NV));

  glAssert(gl->glDepthMask(GL_FALSE));
  glAssert(gl->glDisable(GL_DEPTH_TEST));

  renderQuad->render(gl, objectManager, RenderData());

  glAssert(gl->glDepthMask(GL_TRUE));

  renderTimer.stop();

  float clearTime = clearTimer.waitResult();
  float buildTime = buildTimer.waitResult();
  float renderTime = renderTimer.waitResult();
  qCDebug(channel) << "Clear time" << clearTime << "ms";
  qCDebug(channel) << "Build time" << buildTime << "ms";
  qCDebug(channel) << "Render time" << renderTime << "ms";
}

void HABuffer::setUniforms(std::shared_ptr<ShaderProgram> shader)
{
  shader->setUniform("u_NumRecords", habufferNumRecords);
  shader->setUniform("u_ScreenSz", habufferScreenSize);
  shader->setUniform("u_HashSz", habufferTableSize);
  shader->setUniformAsVec2Array("u_Offsets", offsets, 256);

  shader->setUniform("u_ZNear", zNear);
  shader->setUniform("u_ZFar", zFar);
  shader->setUniform("u_Records", RecordsBuffer);
  shader->setUniform("u_Counts", CountsBuffer);
  shader->setUniform("u_FragmentData", FragmentDataBuffer);
}

void HABuffer::syncAndGetCounts()
{
  glAssert(gl->glMemoryBarrier(GL_ALL_BARRIER_BITS));

  uint numInserted = 1;
  CountsBuffer.getData(&numInserted, sizeof(uint),
                       CountsBuffer.getSize() - sizeof(uint));

  if (numInserted >= habufferNumRecords)
  {
    qCCritical(channel) << "Frame was interrupted:" << numInserted;
  }
  else if (numInserted > habufferNumRecords * 0.8)
  {
    qCWarning(channel) << "inserted" << numInserted << "/"
                       << habufferNumRecords;
  }

  buildTimer.stop();

  displayStatistics("after render");
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
    qCWarning(channel) << label << " habufferCountsSize:" << habufferCountsSize
                       << "<avg:" << avgdepth / static_cast<float>(num)
                       << " max: " << lcounts[habufferCountsSize - 1] << "/"
                       << habufferNumRecords << "(" << rec_percentage << "% "
                       << ">";
  }

  delete[] lcounts;
}

}  // namespace Graphics
