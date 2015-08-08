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

bool TextureManager::initialize(Gl *gl, bool sparse, int maxTextureArrayLevels)
{
  this->gl = gl;
  this->maxTextureArrayLevels = maxTextureArrayLevels;
  this->isSparse = sparse;

  if (maxTextureArrayLevels > 0)
    return true;

  if (sparse)
  {
    glGetIntegerv(GL_MAX_SPARSE_ARRAY_TEXTURE_LAYERS_ARB,
                  &maxTextureArrayLevels);
  }
  else
  {
    glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &maxTextureArrayLevels);
  }

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

TextureAddress TextureManager::getAddressFor(int textureId)
{
  return textures[textureId]->address();
}

Texture2d *
TextureManager::allocateTexture2d(TextureSpaceDescription spaceDescription)
{
  TextureContainer *memArray = nullptr;

  spaceDescription.growToNextPowerOfTwo();

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

}  // namespace Graphics
