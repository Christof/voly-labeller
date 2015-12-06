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

LabelNode::LabelNode(Label label) : label(label)
{
  Importer importer;

  anchorMesh = importer.import("assets/anchor.dae", 0);
  quad = std::make_shared<Graphics::Quad>(":/shader/label.vert",
                                          ":/shader/textureImmediate.frag");

  connector = std::make_shared<Graphics::Connector>(
      ":/shader/pass.vert", ":/shader/colorImmediate.frag",
      Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0));
  connector->color = Eigen::Vector4f(0.75f, 0.75f, 0.75f, 1);
}

LabelNode::~LabelNode()
{
}

void LabelNode::render(Graphics::Gl *gl,
                       std::shared_ptr<Graphics::Managers> managers,
                       RenderData renderData)
{
  if (textureId == -1 || textureText != label.text)
  {
    initialize(gl, managers);
  }

  renderAnchor(gl, managers, renderData);
}

void
LabelNode::renderLabelAndConnector(Graphics::Gl *gl,
                                   std::shared_ptr<Graphics::Managers> managers,
                                   RenderData renderData)
{
  if (textureId == -1 || textureText != label.text)
  {
    initialize(gl, managers);
  }

  renderConnector(gl, managers, renderData);
  renderLabel(gl, managers, renderData);
}

void LabelNode::initialize(Graphics::Gl *gl,
                           std::shared_ptr<Graphics::Managers> managers)
{
  quad->initialize(gl, managers);
  connector->initialize(gl, managers);

  auto image = renderLabelTextToQImage();
  auto textureManager = managers->getTextureManager();
  textureId = textureManager->addTexture(image);
  delete image;

  if (!labelQuad.isInitialized())
  {
    labelQuad = quad->getObjectData();
    labelQuad.setCustomBuffer(sizeof(Graphics::TextureAddress),
                              [textureManager, this](void *insertionPoint)
                              {
      auto textureAddress = textureManager->getAddressFor(textureId);
      std::memcpy(insertionPoint, &textureAddress,
                  sizeof(Graphics::TextureAddress));
    });
  }
}

void LabelNode::renderConnector(Graphics::Gl *gl,
                                std::shared_ptr<Graphics::Managers> managers,
                                RenderData renderData)
{
  Eigen::Vector3f anchorToPosition = labelPosition - label.anchorPosition;
  auto length = anchorToPosition.norm();
  auto rotation = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(),
                                                     anchorToPosition);
  Eigen::Affine3f connectorTransform(
      Eigen::Translation3f(label.anchorPosition) * rotation *
      Eigen::Scaling(length));
  renderData.modelMatrix = connectorTransform.matrix();

  auto shaderId = connector->getObjectData().getShaderProgramId();
  auto shader = managers->getShaderManager()->getShader(shaderId);
  managers->getShaderManager()->bind(shaderId, renderData);
  connector->renderImmediately(gl, managers, renderData);
}

void LabelNode::renderAnchor(Graphics::Gl *gl,
                             std::shared_ptr<Graphics::Managers> managers,
                             RenderData renderData)
{
  Eigen::Affine3f modelTransform(Eigen::Translation3f(label.anchorPosition) *
                                 Eigen::Scaling(0.005f));
  renderData.modelMatrix = modelTransform.matrix();

  anchorMesh->render(gl, managers, renderData);
}

void LabelNode::renderLabel(Graphics::Gl *gl,
                            std::shared_ptr<Graphics::Managers> managers,
                            RenderData renderData)
{
  Eigen::Vector4f anchorNDC = renderData.projectionMatrix *
                              renderData.viewMatrix *
                              toVector4f(label.anchorPosition);
  anchorNDC /= anchorNDC.w();

  Eigen::Vector2f sizeNDC =
      label.size.cwiseQuotient(renderData.windowPixelSize);
  Eigen::Vector4f sizeWorld =
      renderData.projectionMatrix.inverse() *
      Eigen::Vector4f(sizeNDC.x(), sizeNDC.y(), anchorNDC.z(), 1);
  sizeWorld /= sizeWorld.w();

  Eigen::Affine3f labelTransform(
      Eigen::Translation3f(labelPosition) *
      Eigen::Scaling(sizeWorld.x(), sizeWorld.y(), 1.0f));

  labelQuad.modelMatrix = labelTransform.matrix();

  auto shaderId = labelQuad.getShaderProgramId();
  auto shader = managers->getShaderManager()->getShader(shaderId);
  managers->getShaderManager()->bind(shaderId, renderData);

  managers->getObjectManager()->renderImmediately(labelQuad);
}

QImage *LabelNode::renderLabelTextToQImage()
{
  QImage *image = new QImage(512, 128, QImage::Format_ARGB32);
  image->fill(Qt::GlobalColor::transparent);

  QPainter painter;
  painter.begin(image);

  painter.setBrush(QBrush(Qt::GlobalColor::lightGray));
  painter.setPen(Qt::GlobalColor::lightGray);
  painter.drawRoundRect(QRectF(0, 0, 512, 128), 15, 60);

  painter.setPen(Qt::black);
  painter.setFont(QFont("Arial", 72));
  painter.drawText(QRectF(0, 0, 512, 128), Qt::AlignCenter, label.text.c_str());
  painter.end();

  textureText = label.text;

  return image;
}

