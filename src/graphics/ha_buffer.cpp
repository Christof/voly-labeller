#include "./ha_buffer.h"
#include "./shader_program.h"

namespace Graphics
{

HABuffer::HABuffer(Eigen::Vector2i size) : size(size)
{
}

void HABuffer::initialize(Gl *gl)
{
  initializeShadersHash(gl);
  initializeBufferHash(gl);
}

void HABuffer::initializeShadersHash(Gl *gl)
{

  printf("initShaders %d %d\n", size(0), size(1));

  buildShader = std::unique_ptr<ShaderProgram>(new ShaderProgram(
      gl, ":shader/buildHABuffer.vert", ":shader/buildHABuffer.frag"));
  renderShader = std::unique_ptr<ShaderProgram>(new ShaderProgram(
      gl, ":shader/renderHABuffer.vert", ":shader/renderHABuffer.frag"));
  clearShader = std::unique_ptr<ShaderProgram>(new ShaderProgram(
      gl, ":shader/clearHABuffer.vert", ":shader/clearHABuffer.frag"));

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

void HABuffer::initializeBufferHash(Gl *gl)
{
  habufferScreenSize = std::max(size[0], size[1]);
  uint num_records = habufferScreenSize * habufferScreenSize * 8;
  habufferTableSize =
      std::max(habufferScreenSize, (uint)ceil(sqrt((float)num_records)));
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
    FragmentDataBuffer.initialize(gl, habufferNumRecords * sizeof(FragmentData));
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

  /*
  setOrtho(ProjClear, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
  setOrtho(ProjResolve, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
  */
}

}  // namespace Graphics
