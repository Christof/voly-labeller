#include "./attribute_buffer.h"

namespace Graphics
{

AttributeBuffer::AttributeBuffer(int count, int componentCount)
  : m_glID(0), m_Size(count), m_type(c_buf_type),
    componentCount(componentCount), componentSize(4)
{
}

AttributeBuffer::~AttributeBuffer()
{
  if (m_glID == 0)
    return;

  glAssert(gl->glBindBuffer(m_type, 0));

  glAssert(gl->glDeleteBuffers(1, &m_glID));

  m_glID = 0;
  m_Size = 0;
}

int AttributeBuffer::elementSize() const
{
  return componentCount * componentSize;
}

bool AttributeBuffer::isInitialized() const
{
  return (m_glID > 0);
};

GLuint AttributeBuffer::getId() const
{
  return m_glID;
}

uint AttributeBuffer::sizeBytes() const
{
  return (m_Size * elementSize());
}

void AttributeBuffer::initialize(Gl *gl, uint size, GLuint type)
{
  this->gl = gl;
  assert(m_glID == 0);

  m_Size = size;
  m_type = type;

  glAssert(gl->glGenBuffers(1, &m_glID));
  glAssert(gl->glBindBuffer(m_type, m_glID));
  glAssert(gl->glBufferData(m_type, sizeBytes(), NULL, GL_DYNAMIC_DRAW));

  glAssert(gl->glBindBuffer(m_type, 0));
}

void AttributeBuffer::bindAttrib(int attribnum)
{
  assert(m_glID > 0);

  glAssert(gl->glBindBuffer(c_buf_type, m_glID));
  // TODO(SIR): get GL_FLOAT from somewhere else
  glAssert(gl->glVertexAttribPointer(attribnum, componentCount, GL_FLOAT,
                                     GL_FALSE, 0, NULL));
  glAssert(gl->glEnableVertexAttribArray(attribnum));
}
void AttributeBuffer::bindAttribDivisor(int attribnum, int divisor)
{
  assert(m_glID > 0);

  glAssert(gl->glBindBuffer(c_buf_type, m_glID));
  // TODO(SIR): get GL_UNSIGNED_INT from somewhere else
  glAssert(gl->glVertexAttribIPointer(attribnum, componentCount,
                                      GL_UNSIGNED_INT, componentSize, nullptr));
  glAssert(gl->glVertexAttribDivisor(attribnum, divisor));
  glAssert(gl->glEnableVertexAttribArray(attribnum));
}

void AttributeBuffer::bind()
{
  assert(m_glID > 0);

  glAssert(gl->glBindBuffer(m_type, m_glID));
}

void AttributeBuffer::unbind()
{
  glAssert(gl->glBindBuffer(m_type, 0));
}

}  // namespace Graphics
