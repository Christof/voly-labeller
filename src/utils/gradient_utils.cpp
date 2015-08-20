#include "./gradient_utils.h"
#include <QDomDocument>
#include <QPainter>
#include <QFile>
#include <stdexcept>

static QColor loadColor(const QDomElement &elem)
{
  if (elem.tagName() != QLatin1String("colorData"))
    return QColor();

  return QColor(elem.attribute(QLatin1String("r")).toInt(),
                elem.attribute(QLatin1String("g")).toInt(),
                elem.attribute(QLatin1String("b")).toInt(),
                elem.attribute(QLatin1String("a")).toInt());
}

static QGradientStop loadGradientStop(const QDomElement &elem)
{
  if (elem.tagName() != QLatin1String("stopData"))
    return QGradientStop();

  const qreal pos =
      static_cast<qreal>(elem.attribute(QLatin1String("position")).toDouble());
  return qMakePair(pos, loadColor(elem.firstChild().toElement()));
}

static QGradient::Type stringToGradientType(const QString &name)
{
  if (name == QLatin1String("LinearGradient"))
    return QGradient::LinearGradient;
  if (name == QLatin1String("RadialGradient"))
    return QGradient::RadialGradient;
  if (name == QLatin1String("ConicalGradient"))
    return QGradient::ConicalGradient;
  return QGradient::NoGradient;
}

static QGradient::Spread stringToGradientSpread(const QString &name)
{
  if (name == QLatin1String("PadSpread"))
    return QGradient::PadSpread;
  if (name == QLatin1String("RepeatSpread"))
    return QGradient::RepeatSpread;
  if (name == QLatin1String("ReflectSpread"))
    return QGradient::ReflectSpread;
  return QGradient::PadSpread;
}
static QGradient::CoordinateMode
stringToGradientCoordinateMode(const QString &name)
{
  if (name == QLatin1String("LogicalMode"))
    return QGradient::LogicalMode;
  if (name == QLatin1String("StretchToDeviceMode"))
    return QGradient::StretchToDeviceMode;
  if (name == QLatin1String("ObjectBoundingMode"))
    return QGradient::ObjectBoundingMode;
  return QGradient::StretchToDeviceMode;
}

QGradient loadGradientFromDom(const QDomElement &elem)
{
  if (elem.tagName() != QLatin1String("gradientData"))
    return QLinearGradient();

  const QGradient::Type type =
      stringToGradientType(elem.attribute(QLatin1String("type")));
  const QGradient::Spread spread =
      stringToGradientSpread(elem.attribute(QLatin1String("spread")));
  const QGradient::CoordinateMode mode = stringToGradientCoordinateMode(
      elem.attribute(QLatin1String("coordinateMode")));

  QGradient gradient = QLinearGradient();

  if (type == QGradient::LinearGradient)
  {
    QLinearGradient g;
    g.setStart(elem.attribute(QLatin1String("startX")).toDouble(),
               elem.attribute(QLatin1String("startY")).toDouble());
    g.setFinalStop(elem.attribute(QLatin1String("endX")).toDouble(),
                   elem.attribute(QLatin1String("endY")).toDouble());
    gradient = g;
  }
  else if (type == QGradient::RadialGradient)
  {
    QRadialGradient g;
    g.setCenter(elem.attribute(QLatin1String("centerX")).toDouble(),
                elem.attribute(QLatin1String("centerY")).toDouble());
    g.setFocalPoint(elem.attribute(QLatin1String("focalX")).toDouble(),
                    elem.attribute(QLatin1String("focalY")).toDouble());
    g.setRadius(elem.attribute(QLatin1String("radius")).toDouble());
    gradient = g;
  }
  else if (type == QGradient::ConicalGradient)
  {
    QConicalGradient g;
    g.setCenter(elem.attribute(QLatin1String("centerX")).toDouble(),
                elem.attribute(QLatin1String("centerY")).toDouble());
    g.setAngle(elem.attribute(QLatin1String("angle")).toDouble());
    gradient = g;
  }

  QDomElement stopElem = elem.firstChildElement();
  while (!stopElem.isNull())
  {
    QGradientStop stop = loadGradientStop(stopElem);

    gradient.setColorAt(stop.first, stop.second);

    stopElem = stopElem.nextSiblingElement();
  }

  gradient.setSpread(spread);
  gradient.setCoordinateMode(mode);

  return gradient;
}

QGradient GradientUtils::loadGradient(QString path)
{
  QDomDocument doc;
  QFile file(path);
  if (!file.open(QIODevice::ReadOnly))
  {
    throw std::runtime_error("Could not open file" + path.toStdString());
  }
  if (!doc.setContent(&file))
  {
    file.close();
    throw std::runtime_error("Could not read content from file" +
                             path.toStdString());
  }
  file.close();

  return loadGradientFromDom(doc.firstChildElement());
}

QImage GradientUtils::gradientToImage(const QGradient &gradient, QSize size)
{
  QImage image(size, QImage::Format_ARGB32);
  QPainter p(&image);
  p.setCompositionMode(QPainter::CompositionMode_Source);

  const qreal scaleFactor = 0.999999;
  p.scale(scaleFactor, scaleFactor);
  QGradient grad = gradient;
  grad.setCoordinateMode(QGradient::StretchToDeviceMode);
  p.fillRect(QRect(0, 0, size.width(), size.height()), grad);
  p.drawRect(QRect(0, 0, size.width() - 1, size.height() - 1));

  return image;
}

QImage GradientUtils::loadGradientAsImage(QString path, QSize size)
{
  return gradientToImage(loadGradient(path), size);
}

