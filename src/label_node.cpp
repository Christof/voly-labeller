#include "./label_node.h"
#include <QPainter>
#include <QImage>
#include <QPoint>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "./importer.h"
#include "./mesh.h"
#include "./quad.h"
#include "./texture.h"

LabelNode::LabelNode(Label label) : label(label)
{
  Importer importer;

  anchorMesh = importer.import("../assets/anchor.dae", 0);
  quad = std::make_shared<Quad>();
  texture = std::make_shared<Texture>("test.png");
}

LabelNode::~LabelNode()
{
}

void LabelNode::render(Gl *gl, RenderData renderData)
{
  if (!loadedText)
  {
    renderLabelTextToTexture(gl);
    texture->initialize(gl);
  }

  Eigen::Affine3f transform(Eigen::Translation3f(label.anchorPosition) *
                            Eigen::Scaling(0.005f));
  renderData.modelMatrix = transform.matrix();
  anchorMesh->render(gl, renderData);

  Eigen::Affine3f labelTransform(Eigen::Translation3f(1, 0, 1) *
                            Eigen::Scaling(2.0f, 0.5f, 1.0f));
  renderData.modelMatrix = labelTransform.matrix();
  quad->render(gl, renderData, texture);
}

void LabelNode::renderLabelTextToTexture(Gl *gl)
{
  QImage *image = new QImage(512, 128, QImage::Format_ARGB32);
  image->fill(Qt::GlobalColor::transparent);

  QPainter painter;
  painter.begin(image);
  painter.setPen(Qt::blue);
  painter.setFont(QFont("Arial", 72));
  painter.drawText(QRectF(0, 0, 512, 128), Qt::AlignCenter, label.text.c_str());
  painter.end();

  image->save("test.png");

  delete image;
  loadedText = true;
}
