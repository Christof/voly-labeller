#include "./ha_buffer.h"
#include <Eigen/Dense>
#include <QLoggingCategory>
#include <vector>
#include <algorithm>
#include "./shader_program.h"
#include "./screen_quad.h"
#include "./managers.h"
#include "./object_manager.h"
#include "./volume_manager.h"
#include "./volume_data.h"
#include "./texture_manager.h"
#include "./transfer_function_manager.h"
#include "../math/eigen.h"

namespace Graphics
{

QLoggingCategory channel("Graphics.HABuffer");

HABuffer::HABuffer(Eigen::Vector2i size) : size(size)
{
  offsets = new unsigned int[512];
}

HABuffer::~HABuffer()
{
  qCInfo(channel) << "Destructor";
  delete[] offsets;
}

void HABuffer::initialize(Gl *gl, std::shared_ptr<Managers> managers)
{
  this->gl = gl;

  clearQuad = std::make_shared<ScreenQuad>(":/shader/clearHABuffer.vert",
                                           ":/shader/clearHABuffer.frag");
  clearQuad->initialize(gl, managers);

  renderQuad = std::make_shared<ScreenQuad>(":/shader/renderHABuffer.vert",
                                            ":/shader/renderHABuffer.frag");
  renderQuad->initialize(gl, managers);

  initializeBufferHash();

  clearTimer.initialize(gl);
  buildTimer.initialize(gl);
  renderTimer.initialize(gl);
}

void HABuffer::updateNearAndFarPlanes(float nearvalue, float farvalue)
{
  zNear = nearvalue;
  zFar = farvalue;
}

void HABuffer::initializeBufferHash()
{
  habufferScreenSize = std::max(size[0], size[1]);
  uint recordCount = habufferScreenSize * habufferScreenSize * 8;
  habufferTableSize =
      std::max(habufferScreenSize,
               static_cast<uint>(ceil(sqrt(static_cast<float>(recordCount)))));
  tableElementCount = habufferTableSize * habufferTableSize;
  habufferCountsSize = habufferScreenSize * habufferScreenSize + 1;
  qCDebug(channel, "Screen size: %d %d\n# records: %d (%d x %d)\n", size.x(),
          size.y(), tableElementCount, habufferTableSize, habufferTableSize);

  // HA-Buffer records
  if (!recordsBuffer.isInitialized())
    recordsBuffer.initialize(gl, tableElementCount * sizeof(uint) * 2);
  else
    recordsBuffer.resize(tableElementCount * sizeof(uint) * 2);

  if (!countsBuffer.isInitialized())
    countsBuffer.initialize(gl, habufferCountsSize * sizeof(uint));
  else
    countsBuffer.resize(habufferCountsSize * sizeof(uint));

  if (!fragmentDataBuffer.isInitialized())
    fragmentDataBuffer.initialize(gl, tableElementCount * FRAGMENT_DATA_SIZE);
  else
    fragmentDataBuffer.resize(tableElementCount * FRAGMENT_DATA_SIZE);

  // clear counts
  countsBuffer.clear(0);

  gl->glMemoryBarrier(GL_ALL_BARRIER_BITS);

  qCDebug(channel) << "Memory usage:"
                   << ((tableElementCount * sizeof(uint) * 2 +
                        tableElementCount * FRAGMENT_DATA_SIZE +
                        (habufferScreenSize * habufferScreenSize + 1) *
                            sizeof(uint)) /
                       1024) /
                          1024.0f << "MB";
}

void HABuffer::clearAndPrepare(std::shared_ptr<Graphics::Managers> managers)
{
  clearTimer.start();

  for (int i = 0; i < 512; i++)
  {
    offsets[i] = rand() ^ (rand() << 8) ^ (rand() << 16);
    offsets[i] = offsets[i] % habufferTableSize;
  }

  auto clearShader = clearQuad->getShaderProgram();
  clearShader->bind();
  clearShader->setUniform("tableElementCount", tableElementCount);
  clearShader->setUniform("screenSize", habufferScreenSize);
  clearShader->setUniform("records", recordsBuffer);
  clearShader->setUniform("counters", countsBuffer);

  clearQuad->renderImmediately(gl, managers, RenderData());

  if (wireframe)
    gl->glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  // Ensure that all global memory write are done before starting to render
  glAssert(gl->glMemoryBarrier(GL_SHADER_GLOBAL_ACCESS_BARRIER_BIT_NV));

  clearTimer.stop();
  buildTimer.start();

  glAssert(gl->glDisable(GL_CULL_FACE));
}

void HABuffer::begin(std::shared_ptr<ShaderProgram> shader)
{
  // TODO(sirk): re-enable optimization after change to ObjectManager
  // if (lastUsedProgram != shader->getId())
  setUniforms(shader);

  lastUsedProgram = shader->getId();
}

void HABuffer::render(std::shared_ptr<Graphics::Managers> managers,
                      const RenderData &renderData)
{
  if (wireframe)
    gl->glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  gl->glDepthFunc(GL_ALWAYS);
  gl->glDisable(GL_BLEND);

  syncAndGetCounts();

  renderTimer.start();

  auto renderShader = renderQuad->getShaderProgram();
  renderShader->bind();

  renderShader->setUniform("screenSize", habufferScreenSize);
  renderShader->setUniform("tableSize", habufferTableSize);
  renderShader->setUniformAsVec2Array("offsets", offsets, 256);
  renderShader->setUniform("records", recordsBuffer);
  renderShader->setUniform("counters", countsBuffer);
  renderShader->setUniform("fragmentData", fragmentDataBuffer);

  Eigen::Matrix4f inverseViewMatrix = renderData.viewMatrix.inverse();
  renderShader->setUniform("inverseViewMatrix", inverseViewMatrix);
  renderShader->setUniform("projectionMatrix", renderData.projectionMatrix);

  Eigen::Vector3f textureAtlasSize =
      managers->getVolumeManager()->getVolumeAtlasSize().cast<float>();
  renderShader->setUniform("textureAtlasSize", textureAtlasSize);
  Eigen::Vector3f sampleDistance =
      Eigen::Vector3f(0.49f, 0.49f, 0.49f).cwiseQuotient(textureAtlasSize);
  renderShader->setUniform("sampleDistance", sampleDistance);
  renderShader->setUniform(
      "transferFunctionWidth",
      managers->getTransferFunctionManager()->getTextureWidth());

  setLayeringUniforms(renderShader, renderData);

  ObjectData &objectData = renderQuad->getObjectDataReference();
  auto volumeData = managers->getVolumeManager()->getBufferData();
  int volumeDataSize = volumeData.size() * sizeof(Graphics::VolumeData);

  objectData.setCustomBuffer(volumeDataSize,
                             [volumeData, volumeDataSize](void *insertionPoint)
                             {
    std::memcpy(insertionPoint, volumeData.data(), volumeDataSize);
  });

  // Ensure that all global memory write are done before resolving
  glAssert(gl->glMemoryBarrier(GL_SHADER_GLOBAL_ACCESS_BARRIER_BIT_NV));

  renderQuad->renderImmediately(gl, managers, renderData);

  renderTimer.stop();

  float clearTime = clearTimer.waitResult();
  float buildTime = buildTimer.waitResult();
  float renderTime = renderTimer.waitResult();
  qCDebug(channel) << "Clear time" << clearTime << "ms";
  qCDebug(channel) << "Build time" << buildTime << "ms";
  qCDebug(channel) << "Render time" << renderTime << "ms";

  gl->glDepthFunc(GL_LESS);
  gl->glEnable(GL_BLEND);
}

void HABuffer::setLayerZValues(std::vector<float> layerZValues)
{
  this->layerZValues = layerZValues;
}

void HABuffer::setUniforms(std::shared_ptr<ShaderProgram> shader)
{
  shader->setUniform("tableElementCount", tableElementCount);
  shader->setUniform("screenSize", habufferScreenSize);
  shader->setUniform("tableSize", habufferTableSize);
  shader->setUniformAsVec2Array("offsets", offsets, 256);

  shader->setUniform("near", zNear);
  shader->setUniform("far", zFar);
  shader->setUniform("records", recordsBuffer);
  shader->setUniform("counters", countsBuffer);
  shader->setUniform("fragmentData", fragmentDataBuffer);
}

void HABuffer::setLayeringUniforms(std::shared_ptr<ShaderProgram> renderShader,
                                   const RenderData &renderData)
{
  std::vector<Eigen::Vector4f> layerPlanes;
  for (auto &layerZValue : layerZValues)
  {
    Eigen::Vector4f probePointNdc(0.0, 0.0, layerZValue, 1);
    Eigen::Vector3f probePointEye =
        project(renderData.projectionMatrix.inverse(), probePointNdc);
    layerPlanes.push_back(Eigen::Vector4f(0, 0, 1, -probePointEye.z()));
  }

  std::sort(layerPlanes.begin(), layerPlanes.end(),
            [](const Eigen::Vector4f &left, const Eigen::Vector4f &right)
            {
    return left.w() < right.w();
  });

  int layerCount = static_cast<int>(layerZValues.size());
  int planeCount = static_cast<int>(layerPlanes.size());
  renderShader->setUniform("layerCount", layerCount);
  renderShader->setUniformAsVec4Array("layerPlanes", layerPlanes.data(),
                                      planeCount);
  renderShader->setUniform("planeCount", planeCount);
  renderShader->setUniformAsFloatArray("planesZValuesNdc", layerZValues.data(),
                                       layerCount);
}

void HABuffer::syncAndGetCounts()
{
  glAssert(gl->glMemoryBarrier(GL_ALL_BARRIER_BITS));

  uint numInserted = 1;
  countsBuffer.getData(&numInserted, sizeof(uint),
                       countsBuffer.getSize() - sizeof(uint));

  if (numInserted >= tableElementCount)
  {
    qCCritical(channel) << "Frame was interrupted:" << numInserted;
  }
  else if (numInserted > tableElementCount * 0.8)
  {
    qCWarning(channel) << "inserted" << numInserted << "/" << tableElementCount;
  }

  buildTimer.stop();

  displayStatistics("after render");
}

void HABuffer::displayStatistics(const char *label)
{
  uint *lcounts = new uint[habufferCountsSize];

  countsBuffer.getData(lcounts, countsBuffer.getSize());

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
                          static_cast<double>(tableElementCount) * 100.0;

  if (rec_percentage > 80.0)
  {
    qCWarning(channel) << label << " habufferCountsSize:" << habufferCountsSize
                       << "<avg:" << avgdepth / static_cast<float>(num)
                       << " max: " << lcounts[habufferCountsSize - 1] << "/"
                       << tableElementCount << "(" << rec_percentage << "% "
                       << ">";
  }

  delete[] lcounts;
}

}  // namespace Graphics
