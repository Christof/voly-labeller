#include "./texture_manager.h"
#include <QLoggingCategory>
#include <vector>
#include <map>
#include <string>
#include <cassert>
#include "./texture_container.h"
#include "./texture2d.h"
#include "./gl.h"

namespace Graphics
{

QLoggingCategory tmChan("Graphics.TextureManager");

TextureManager::~TextureManager()
{
  for (auto texture : textures)
    delete texture;

  textures.clear();

  shutdown();
}

int TextureManager::addTexture(std::string path)
{
  int id = textures.size();

  textures.push_back(newTexture2d(path));

  return id;
}

int TextureManager::addTexture(QImage *image)
{
  int id = textures.size();

  textures.push_back(newTexture2d(image));

  return id;
}

int TextureManager::addTexture(float *data, int width, int height)
{
  int id = textures.size();

  textures.push_back(newTexture2d(data, width, height));

  return id;
}

unsigned int TextureManager::add3dTexture(Eigen::Vector3i size, float *data)
{
  auto textureTarget = GL_TEXTURE_3D;
  unsigned int texture = 0;

  glAssert(gl->glGenTextures(1, &texture));
  glAssert(gl->glBindTexture(textureTarget, texture));
  glAssert(gl->glTexImage3D(textureTarget, 0, GL_R32F, size.x(), size.y(),
                            size.z(), 0, GL_RED, GL_FLOAT, data));

  glTexParameterf(textureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameterf(textureTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(textureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(textureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexParameteri(textureTarget, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

  return texture;
}

Texture2d *
TextureManager::newTexture2d(TextureSpaceDescription spaceDescription)
{
  Texture2d *texture = allocateTexture2d(spaceDescription);
  texture->commit();

  return texture;
}

Texture2d *TextureManager::newTexture2d(std::string path)
{
  auto image = new QImage(path.c_str());
  auto texture = newTexture2d(image);

  delete image;

  return texture;
}

Texture2d *TextureManager::newTexture2d(QImage *image)
{
  try
  {
    auto internalformat = GL_RGBA8;
    auto format = GL_BGRA;
    auto type = GL_UNSIGNED_BYTE;

    Texture2d *texture = allocateTexture2d(TextureSpaceDescription(
        1, internalformat, image->width(), image->height()));
    texture->commit();

    texture->texSubImage2D(0, 0, 0, image->width(), image->height(), format,
                           type, image->bits());

    return texture;
  }
  catch (std::exception &error)
  {
    qCCritical(tmChan) << "Error loading texture: " << error.what();
    throw;
  }
}

Texture2d *TextureManager::newTexture2d(float *data, int width, int height)
{
  auto internalformat = GL_RGBA32F;
  auto format = GL_RGBA;
  auto type = GL_FLOAT;

  Texture2d *texture = allocateTexture2d(
      TextureSpaceDescription(1, internalformat, width, height));
  texture->commit();

  texture->texSubImage2D(0, 0, 0, width, height, format, type, data);

  return texture;
}

bool TextureManager::initialize(Gl *gl, bool sparse, int maxTextureArrayLevels)
{
  this->gl = gl;
  this->maxTextureArrayLevels = maxTextureArrayLevels;
  this->isSparse = sparse;

  if (maxTextureArrayLevels > 0)
    return true;

  auto layersKey = sparse ? GL_MAX_SPARSE_ARRAY_TEXTURE_LAYERS_ARB
                          : GL_MAX_ARRAY_TEXTURE_LAYERS;
  glGetIntegerv(layersKey, &maxTextureArrayLevels);

  return true;
}

void TextureManager::shutdown()
{
  for (auto containIt = textureContainers.begin();
       containIt != textureContainers.end(); ++containIt)
  {
    for (auto ptrIt = containIt->second.begin();
         ptrIt != containIt->second.end(); ++ptrIt)
    {
      delete *ptrIt;
    }
  }

  textureContainers.clear();
}

Texture2d *TextureManager::getTextureFor(int textureId)
{
  if (static_cast<int>(textures.size()) <= textureId)
    throw std::invalid_argument("The given textureId " +
                                std::to_string(textureId) + "cannot be found");

  return textures[textureId];
}

TextureAddress TextureManager::getAddressFor(int textureId)
{
  return getTextureFor(textureId)->address();
}

Texture2d *
TextureManager::allocateTexture2d(TextureSpaceDescription spaceDescription)
{
  TextureContainer *memArray = nullptr;

  int virtualPageSizeX = get2DVirtualPageSizeX(spaceDescription.internalFormat);
  int virtualPageSizeY = get2DVirtualPageSizeY(spaceDescription.internalFormat);
  qCInfo(tmChan) << "Virtual page size: " << virtualPageSizeX << "/"
                 << virtualPageSizeY;
  spaceDescription.growToValidSize(virtualPageSizeX, virtualPageSizeY);

  auto arrayIt = textureContainers.find(spaceDescription);
  if (arrayIt == textureContainers.end())
  {
    textureContainers[spaceDescription] = std::vector<TextureContainer *>();
    arrayIt = textureContainers.find(spaceDescription);
    assert(arrayIt != textureContainers.end());
  }

  for (auto it = arrayIt->second.begin(); it != arrayIt->second.end(); ++it)
  {
    if ((*it)->hasRoom())
    {
      memArray = (*it);
      break;
    }
  }

  if (memArray == nullptr)
  {
    memArray = new TextureContainer(gl, isSparse, spaceDescription,
                                    maxTextureArrayLevels);
    arrayIt->second.push_back(memArray);
  }

  assert(memArray);
  return new Texture2d(memArray, memArray->virtualAlloc());
}

int TextureManager::get2DVirtualPageSizeX(int internalFormat)
{
  return getInternalFormat(GL_TEXTURE_2D, internalFormat, 0x9195);
}

int TextureManager::get2DVirtualPageSizeY(int internalFormat)
{
  return getInternalFormat(GL_TEXTURE_2D, internalFormat, 0x9196);
}

int TextureManager::getInternalFormat(int target, int internalFormat,
                                      int parameterName)
{
  int result = -1;
  gl->glGetInternalformativ(target, internalFormat, parameterName, 1, &result);
  return result;
}

}  // namespace Graphics
