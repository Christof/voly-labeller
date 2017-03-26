#include "./transfer_function_parser.h"
#include <QFile>
#include <QGradient>
#include <QGradientStops>
#include <QXmlSimpleReader>
#include <QXmlInputSource>
#include <QMetaEnum>
#include <QException>
#include <QLoggingCategory>

QLoggingCategory tfpChan("Utils.TransferFunctionParser");

/**
 * \brief Helper class to parse QGradients by implementing a QXmlDefaultHandler
 *
 */
class QGradientContentHandler : public QXmlDefaultHandler
{
 private:
  QGradient instance;

  QGradientStops stops;
  double stopPosition;
  QColor stopColor;

 public:
  QGradientContentHandler() : QXmlDefaultHandler()
  {
  }

  ~QGradientContentHandler()
  {
  }

  QGradient getQGradientInstance()
  {
    return instance;
  }

  bool fatalError(const QXmlParseException &exception)
  {
    qCCritical(tfpChan) << "Fatal error on line" << exception.message();

    return false;
  }

  QGradient::Spread parseSpread(QString spreadString)
  {
    if (spreadString.compare("PadSpread") == 0)
      return QGradient::Spread::PadSpread;
    if (spreadString.compare("ReflectSpread") == 0)
      return QGradient::Spread::ReflectSpread;
    if (spreadString.compare("RepeatSpread") == 0)
      return QGradient::Spread::RepeatSpread;
    else
      throw new QString("Spread value not supported!");
  }

  QGradient::CoordinateMode parseCoordinateMode(QString coordinateModeString)
  {
    if (coordinateModeString.compare("LogicalMode") == 0)
      return QGradient::CoordinateMode::LogicalMode;
    else if (coordinateModeString.compare("StretchToDeviceMode") == 0)
      return QGradient::CoordinateMode::StretchToDeviceMode;
    else if (coordinateModeString.compare("ObjectBoundingMode") == 0)
      return QGradient::CoordinateMode::ObjectBoundingMode;
    else
      throw new QString("CoordinateMode value not supported!");
  }

  void parseGradientAttributes(const QXmlAttributes &atts)
  {
    if (atts.value("type").compare(("LinearGradient")) == 0)
    {
      auto startX = atts.value("startX").toDouble();
      auto startY = atts.value("startY").toDouble();
      auto endX = atts.value("endX").toDouble();
      auto endY = atts.value("endY").toDouble();

      instance = QLinearGradient(startX, startY, endX, endY);
    }
    else if (atts.value("type").compare("RadialGradient") == 0)
    {
      auto centerX = atts.value("centerX").toDouble();
      auto centerY = atts.value("centerY").toDouble();
      auto radius = atts.value("radius").toDouble();
      auto fX = atts.value("fX").toDouble();
      auto fY = atts.value("fY").toDouble();

      instance = QRadialGradient(centerX, centerY, radius, fX, fY);
    }
    else if (atts.value("type").compare("ConicalGradient") == 0)
    {
      auto centerX = atts.value("centerX").toDouble();
      auto centerY = atts.value("centerY").toDouble();
      auto startAngle = atts.value("startAngle").toDouble();

      instance = QConicalGradient(centerX, centerY, startAngle);
    }
    else
      throw new QString("gradient type not supported!");

    instance.setSpread(parseSpread(atts.value("spread")));
    instance.setCoordinateMode(
        parseCoordinateMode(atts.value("coordinateMode")));

    // reset stops here
    stops.clear();
  }

  void parseStopData(const QXmlAttributes &atts)
  {
    stopPosition = atts.value("position").toDouble();
  }

  void parseColorData(const QXmlAttributes &atts)
  {
    stopColor.setRed(atts.value("r").toInt());
    stopColor.setGreen(atts.value("g").toInt());
    stopColor.setBlue(atts.value("b").toInt());
    stopColor.setAlpha(atts.value("a").toInt());
  }

  bool startElement(const QString &namespaceURI, const QString &localName,
                    const QString &qName, const QXmlAttributes &atts)
  {
    Q_UNUSED(namespaceURI);
    Q_UNUSED(qName);
    qCDebug(tfpChan) << "Read Start Tag" << localName;

    if (localName.compare("gradientData") == 0)
      parseGradientAttributes(atts);
    if (localName.compare("stopData") == 0)
      parseStopData(atts);
    if (localName.compare("colorData") == 0)
      parseColorData(atts);

    return true;
  }

  bool endElement(const QString &namespaceURI, const QString &localName,
                  const QString &qName)
  {
    Q_UNUSED(namespaceURI);
    Q_UNUSED(qName);
    qCDebug(tfpChan) << "Read End Element Tag";

    if (localName.compare("gradientData") == 0)
    {
      qCDebug(tfpChan) << "Setting Stops with size:" << stops.size();
      instance.setStops(stops);
    }
    else if (localName.compare("colorData") == 0)
      stops.append(qMakePair(stopPosition, stopColor));

    return true;
  }
};

TransferFunctionParser::TransferFunctionParser(QString path)
{
  qCInfo(tfpChan) << "Create parser for" << path;

  xmlReader = new QXmlSimpleReader();
  source = new QXmlInputSource(new QFile(path));
  handler = new QGradientContentHandler();
  xmlReader->setContentHandler(handler);
  xmlReader->setErrorHandler(handler);
}

QGradient TransferFunctionParser::parse()
{
  bool ok = xmlReader->parse(source);

  if (!ok)
  {
    qCCritical(tfpChan) << "Parsing failed!";
  }

  return handler->getQGradientInstance();
}
