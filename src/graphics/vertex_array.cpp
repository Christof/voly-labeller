#include "./vertex_array.h"
#include <vector>

namespace Graphics
{

VertexArray::VertexArray(Gl *gl, GLenum primitiveMode,
                         int positionElementsCount)
  : gl(gl), primitiveMode(primitiveMode),
    positionElementsCount(positionElementsCount)
{
  gl->glGenVertexArrays(1, &vertexArrayId);
}

VertexArray::~VertexArray()
{
  for (unsigned int cnt = 0; cnt < data.size(); cnt++)
    delete data[cnt];

  gl->glBindVertexArray(0);
  gl->glDeleteVertexArrays(1, &vertexArrayId);
}

void VertexArray::addStream(std::vector<float> stream, int elementSize)
{
  data.push_back(new VertexBuffer(gl, stream, elementSize));
}

void VertexArray::draw()
{
  gl->glBindVertexArray(vertexArrayId);

  for (unsigned int i = 0; i < data.size(); i++)
  {
    gl->glEnableVertexAttribArray(i);
    data[i]->bind();
    gl->glVertexAttribPointer(i, data[i]->getElementSize(), GL_FLOAT, GL_FALSE,
                              0, nullptr);
  }

  if (data.size() > 0)
    gl->glDrawArrays(primitiveMode, 0,
                     static_cast<uint>(data[0]->getSize()) / positionElementsCount);

  for (unsigned int cnt = 0; cnt < data.size(); cnt++)
    gl->glDisableVertexAttribArray(cnt);

  gl->glBindBuffer(GL_ARRAY_BUFFER, 0);
  gl->glBindVertexArray(0);
}

}  // namespace Graphics

