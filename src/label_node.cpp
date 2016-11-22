#include "./label_node.h"
#include <QPainter>
#include <QImage>
#include <QPoint>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "./graphics/texture_address.h"
#include "./graphics/shader_program.h"
#include "./importer.h"
#include "./math/eigen.h"
#include "./math/utils.h"
#include "./labelling/labeller_frame_data.h"

const Eigen::Vector4f color = { 0.85f, 0.85f, 0.85f, 1 };

LabelNode::LabelNode(Label label) : label(label)
{
  isVisible = false;
  Importer importer;

  anchorMesh = importer.import("assets/models/anchor.dae", 0);
  quad = std::make_shared<Graphics::Quad>(":/shader/label.vert",
                                          ":/shader/label.frag");

  connector = std::make_shared<Graphics::Connector>(
      ":/shader/pass.vert", ":/shader/connector.frag", Eigen::Vector3f(0, 0, 0),
      Eigen::Vector3f(1, 0, 0));
  connector->color = color;
}

LabelNode::~LabelNode()
{
}

void LabelNode::render(Graphics::Gl *gl,
                       std::shared_ptr<Graphics::Managers> managers,
                       RenderData renderData)
{
  isOutsideFieldOfView =
      !label.isAnchorInsideFieldOfView(renderData.viewProjectionMatrix);
  if (isOutsideFieldOfView)
    return;

  if (textureId == -1 || textureText != label.text || labelSize != label.size)
  {
    initialize(gl, managers);
  }

  if (isVisible && isAnchorVisible)
    renderAnchor(gl, managers, renderData);
}

void
LabelNode::renderLabelAndConnector(Graphics::Gl *gl,
                                   std::shared_ptr<Graphics::Managers> managers,
                                   RenderData renderData)
{
  if (isOutsideFieldOfView)
  {
    if (isVisible)
      setIsVisible(false);
  }

  if (textureId == -1 || textureText != label.text)
  {
    initialize(gl, managers);
  }

  if (isVisible || timeSinceIsVisibleChanged < fadeTime)
  {
    updateAlphaValue();

    if (alpha > 0)
    {
      gl->glStencilFunc(GL_ALWAYS, 255, 0xFF);
      gl->glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
      gl->glStencilMask(0xFF);
      renderLabel(gl, managers, renderData);

      gl->glStencilFunc(GL_EQUAL, 0, 0xFF);
      gl->glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
      renderConnector(gl, managers, renderData);
    }

    timeSinceIsVisibleChanged += renderData.frameTime;
  }
}

void LabelNode::setIsVisible(bool isVisible)
{
  if (isVisible && isOutsideFieldOfView)
    return;

  if (this->isVisible != isVisible)
    timeSinceIsVisibleChanged = 0.0f;

  this->isVisible = isVisible;
}

void LabelNode::initialize(Graphics::Gl *gl,
                           std::shared_ptr<Graphics::Managers> managers)
{
  quad->initialize(gl, managers);
  connector->initialize(gl, managers);

  auto image = renderLabelTextToQImage();
  auto textureManager = managers->getTextureManager();
  if (textureId >= 0)
    textureManager->free(textureId);

  textureId = textureManager->addTexture(image);
  delete image;

  if (!labelQuad.isInitialized())
  {
    labelQuad = quad->getObjectData();
    labelQuad.setCustomBufferFor<Graphics::TextureAddress>(
        1, [textureManager, this]()
        {
          auto textureAddress = textureManager->getAddressFor(textureId);
          textureAddress.reserved = this->layerIndex;
          return textureAddress;
        });
    labelQuad.setCustomBufferFor<float>(2, &this->alpha);
  }

  if (!labelConnector.isInitialized())
  {
    labelConnector = connector->getObjectData();
    labelConnector.setCustomBufferFor(1, &this->layerIndex);
    labelConnector.setCustomBufferFor(2, &this->alpha);
  }
}

void LabelNode::renderConnector(Graphics::Gl *gl,
                                std::shared_ptr<Graphics::Managers> managers,
                                RenderData renderData)
{
  gl->glLineWidth(2.0f);
  Eigen::Vector3f anchorToPosition = labelPosition - label.anchorPosition;
  auto length = anchorToPosition.norm();
  auto rotation = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(),
                                                     anchorToPosition);
  Eigen::Affine3f connectorTransform(
      Eigen::Translation3f(label.anchorPosition) * rotation *
      Eigen::Scaling(length));
  labelConnector.modelMatrix = connectorTransform.matrix();

  auto shaderId = labelConnector.getShaderProgramId();
  auto shader = managers->getShaderManager()->getShader(shaderId);
  managers->getShaderManager()->bind(shaderId, renderData);
  managers->getObjectManager()->renderImmediately(labelConnector);
}

void LabelNode::renderAnchor(Graphics::Gl *gl,
                             std::shared_ptr<Graphics::Managers> managers,
                             RenderData renderData)
{
  anchorNDC = project(renderData.viewProjectionMatrix, label.anchorPosition);

  float sizeNDC = anchorSize / renderData.windowPixelSize.x();

  Eigen::Vector3f sizeWorld =
      calculateWorldScale(Eigen::Vector4f(sizeNDC, sizeNDC, anchorNDC.z(), 1),
                          renderData.projectionMatrix);

  Eigen::Affine3f anchorTransform(Eigen::Translation3f(label.anchorPosition) *
                                  Eigen::Scaling(sizeWorld.x()));
  renderData.modelMatrix = anchorTransform.matrix();

  anchorMesh->render(gl, managers, renderData);
}

void LabelNode::renderLabel(Graphics::Gl *gl,
                            std::shared_ptr<Graphics::Managers> managers,
                            RenderData renderData)
{
  Eigen::Vector2f sizeNDC =
      label.size.cwiseQuotient(renderData.windowPixelSize);

  Eigen::Affine3f labelTransform(
      Eigen::Translation3f(labelPositionNDC) *
      Eigen::Scaling(sizeNDC.x(), sizeNDC.y(), 1.0f));

  labelQuad.modelMatrix = labelTransform.matrix();

  auto shaderId = labelQuad.getShaderProgramId();
  managers->getShaderManager()->bind(shaderId, renderData);
  managers->getShaderManager()->getShader(shaderId)->setUniform(
      "maxTextureCoord", maxTextureCoordinates);

  managers->getObjectManager()->renderImmediately(labelQuad);
}

QImage *LabelNode::renderLabelTextToQImage()
{
  int width = 4 * label.size.x();
  int height = 4 * label.size.y();
  int textureWidth = Math::computeNextPowerOfTwo(width);
  int textureHeight = Math::computeNextPowerOfTwo(height);
  QImage *image =
      new QImage(textureWidth, textureHeight, QImage::Format_RGBA8888);
  QColor c = QColor::fromRgbF(color.x(), color.y(), color.z(), color.w());
  image->fill(c);

  QPainter painter;
  painter.begin(image);

  painter.setPen(Qt::black);
  painter.setFont(QFont("Arial", 72));
  painter.drawText(QRectF(0, 0, width, height), Qt::AlignCenter,
                   label.text.c_str());
  painter.end();

  textureText = label.text;
  labelSize = label.size;
  maxTextureCoordinates =
      Eigen::Vector2f(static_cast<float>(width) / textureWidth,
                      static_cast<float>(height) / textureHeight);

  return image;
}

void LabelNode::updateAlphaValue()
{
  if (timeSinceIsVisibleChanged < fadeTime)
  {
    float percent = timeSinceIsVisibleChanged / fadeTime;
    alpha = isVisible ? percent : (1.0f - percent);
  }
  else
  {
    alpha = isVisible ? 1.0f : 0.0f;
  }
}
