#include "./buffer_manager.h"

namespace Graphics
{

BufferManager::BufferManager()
  : positionBuffer(3, sizeof(float), GL_FLOAT),
    normalBuffer(3, sizeof(float), GL_FLOAT),
    colorBuffer(4, sizeof(float), GL_FLOAT),
    texCoordBuffer(2, sizeof(float), GL_FLOAT),
    drawIdBuffer(1, sizeof(uint), GL_UNSIGNED_INT),
    indexBuffer(1, sizeof(uint), GL_UNSIGNED_INT), vertexBufferManager(0),
    indexBufferManager(0)
{
}

BufferManager::~BufferManager()
{
  if (!gl)
    return;

  glAssert(gl->glDeleteVertexArrays(1, &vertexArrayId));

  /*
  // clear textures
  for (auto texture = m_Textures.begin(); texture != m_Textures.end();
  texture++)
  {
    delete (*texture);
    *texture = nullptr;
  }
  m_Textures.clear();

  m_TextureManager.Shutdown();
  */
}

void BufferManager::initialize(Gl *gl, uint maxobjects, uint buffersize)
{
  this->gl = gl;
  // gl_assert(m_VertexArrayID == 0);

  // m_TextureManager.Init(true, 8);

  vertexBufferManager = BufferHoleManager(buffersize);
  indexBufferManager = BufferHoleManager(buffersize);

  glAssert(gl->glGenVertexArrays(1, &vertexArrayId));
  glAssert(gl->glBindVertexArray(vertexArrayId));

  positionBuffer.initialize(gl, buffersize);
  normalBuffer.initialize(gl, buffersize);
  colorBuffer.initialize(gl, buffersize);
  texCoordBuffer.initialize(gl, buffersize);

  indexBuffer.initialize(gl, buffersize, GL_ELEMENT_ARRAY_BUFFER);

  // initialize DrawID buffer - ascending ids
  drawIdBuffer.initialize(gl, maxobjects);
  std::vector<uint> drawids;
  uint idval = 0;
  drawids.resize(maxobjects);

  for_each(drawids.begin(), drawids.end(),
           [&idval](std::vector<uint>::value_type &v)
           {
    v = idval++;
  });
  // std::cout << "drawIDs:" << drawids << std::endl;

  drawIdBuffer.setData(drawids);

  // bind per vertex attibutes to vertex array
  positionBuffer.bindAttrib(0);
  normalBuffer.bindAttrib(1);
  colorBuffer.bindAttrib(2);
  texCoordBuffer.bindAttrib(3);
  drawIdBuffer.bindAttribDivisor(4, 1);

  glAssert(gl->glBindVertexArray(0));

  /*
  m_CommandsBuffer.init(3 * maxobjects, c_createFlags, c_mapFlags);
  m_TransformBuffer.init(3 * maxobjects, c_createFlags, c_mapFlags);
  m_TexAddressBuffer.init(3 * maxobjects, c_createFlags, c_mapFlags);
  */
}

int BufferManager::addObject(const std::vector<float> &vertices,
                             const std::vector<float> &normals,
                             const std::vector<float> &colors,
                             const std::vector<float> &texCoords,
                             const std::vector<uint> &indices)
{

  assert(vertices.size() /
             static_cast<float>(positionBuffer.getComponentCount()) ==
         normals.size() / static_cast<float>(normalBuffer.getComponentCount()));
  assert(vertices.size() /
             static_cast<float>(positionBuffer.getComponentCount()) ==
         colors.size() / static_cast<float>(colorBuffer.getComponentCount()));
  assert(vertices.size() /
             static_cast<float>(positionBuffer.getComponentCount()) ==
         texCoords.size() /
             static_cast<float>(texCoordBuffer.getComponentCount()));

  const uint vertexCount = vertices.size() / positionBuffer.getComponentCount();

  // try to reserve buffer storage for objects

  uint vertexBufferOffset;
  uint indexBufferOffset;

  bool reserve_success =
      vertexBufferManager.reserve(vertexCount, vertexBufferOffset);
  if (reserve_success)
  {
    reserve_success =
        indexBufferManager.reserve(indices.size(), indexBufferOffset);
    if (!reserve_success)
    {
      vertexBufferManager.release(vertexBufferOffset);
    }
  }

  if (!reserve_success)
    return -1;

  // fill buffers
  positionBuffer.setData(vertices, vertexBufferOffset);
  normalBuffer.setData(normals, vertexBufferOffset);
  colorBuffer.setData(colors, vertexBufferOffset);
  texCoordBuffer.setData(texCoords, vertexBufferOffset);

  indexBuffer.setData(indices, indexBufferOffset);

  ObjectData object;
  object.VertexOffset = vertexBufferOffset;
  object.VertexSize = vertexCount;
  object.IndexOffset = indexBufferOffset;
  object.IndexSize = indices.size();
  object.TextureAddress = { 0, 0.0f, 0, 1.0f, 1.0f };
  object.transform = Eigen::Matrix4f::Identity();

  int objectId = objectCount++;

  objects.insert(std::make_pair(objectId, object));

  return objectId;
}

}  // namespace Graphics
