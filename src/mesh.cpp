#include "./mesh.h"
#include "./gl_assert.h"
#include <QOpenGLFunctions_4_3_Core>
#include <QDebug>

Mesh::Mesh(QOpenGLFunctions_4_3_Core *gl, aiMesh *mesh, aiMaterial *material)
  : gl(gl), shaderProgram(gl, ":shader/phong.vert", ":shader/phong.frag")
{
  vertexCount = mesh->mNumFaces * 3;

  for (unsigned int i = 0; i < material->mNumProperties; ++i)
  {
    auto property = material->mProperties[i];
    std::cout << property->mKey.C_Str() << ": " << property->mType << "|"
              << property->mDataLength << std::endl;
  }

  ambientColor = loadVector4FromMaterial("$clr.ambient", material);
  diffuseColor = loadVector4FromMaterial("$clr.diffuse", material);
  specularColor = loadVector4FromMaterial("$clr.specular", material);
  shininess = loadFloatFromMaterial("$mat.shininess", material);

  std::cout << "diffuse: " << diffuseColor << " ambient: " << ambientColor
            << " specular: " << specularColor << " shininess: " << shininess
            << std::endl;

  auto positionData = new float[mesh->mNumFaces * 3 * 3];
  float *positionInsertPoint = positionData;
  auto normalData = new float[mesh->mNumFaces * 3 * 3];
  float *normalInsertPoint = normalData;

  for (unsigned int i = 0; i < mesh->mNumFaces; i++)
  {
    const aiFace &face = mesh->mFaces[i];

    for (int j = 0; j < 3; j++)
    {

      aiVector3D pos = mesh->mVertices[face.mIndices[j]];
      memcpy(positionInsertPoint, &pos, sizeof(float) * 3);
      positionInsertPoint += 3;

      aiVector3D normal = mesh->mNormals[face.mIndices[j]];
      memcpy(normalInsertPoint, &normal, sizeof(float) * 3);
      normalInsertPoint += 3;
    }
  }

  vertexArrayObject.create();
  vertexArrayObject.bind();

  shaderProgram.bind();

  createBuffer(positionData, "vertexPosition", 3, vertexCount);
  createBuffer(normalData, "vertexNormal", 3, vertexCount);
}

Mesh::~Mesh()
{
}

Eigen::Vector4f Mesh::loadVector4FromMaterial(const char *key,
                                              aiMaterial *material)
{
  float values[4];
  unsigned int size = 4;
  if (material->Get(key, 0, 0, values, &size) != 0)
  {
    qCritical() << "Could not load " << key << " from material";
    exit(1);
  }

  return Eigen::Vector4f(values[0], values[1], values[2], values[3]);
}

float Mesh::loadFloatFromMaterial(const char *key, aiMaterial *material)
{
  float result;
  if (material->Get(key, 0, 0, result) != 0)
  {
    qCritical() << "Could not load " << key << " from material";
    exit(1);
  }

  return result;
}

void Mesh::createBuffer(float *data, std::string usage, int perVertexElements,
                        int numberOfVertices)
{
  QOpenGLBuffer buffer(QOpenGLBuffer::VertexBuffer);
  buffer.create();
  buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
  buffer.bind();
  buffer.allocate(data, numberOfVertices * perVertexElements * sizeof(float));
  glCheckError();

  shaderProgram.enableAndSetAttributes(usage, perVertexElements);

  buffers.push_back(buffer);
}

void Mesh::prepareShaderProgram()
{
}

void Mesh::render(Eigen::Matrix4f projection, Eigen::Matrix4f view)
{
  shaderProgram.bind();

  Eigen::Matrix4f modelViewProjection = projection * view;
  shaderProgram.setUniform("viewProjectionMatrix", modelViewProjection);
  shaderProgram.setUniform("ambientColor", ambientColor);
  shaderProgram.setUniform("diffuseColor", diffuseColor);
  shaderProgram.setUniform("specularColor", specularColor);
  shaderProgram.setUniform("cameraDirection",
             Eigen::Vector3f(view(2, 0), view(2, 1), view(2, 2)));
  shaderProgram.setUniform("shininess", shininess);

  vertexArrayObject.bind();

  glAssert(gl->glDrawArrays(GL_TRIANGLES, 0, vertexCount));
}

