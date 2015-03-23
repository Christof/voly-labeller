#include "./scxml_importer.h"
#include <QXmlStreamReader>
#include <QFile>
#include <iostream>

ScxmlImporter::ScxmlImporter(QUrl url)
{
  QFile file(url.toLocalFile());
  std::cout << file.exists() << std::endl;
  file.open(QFile::OpenModeFlag::ReadOnly);

  QXmlStreamReader reader(file.readAll());
  while (!reader.atEnd() && !reader.hasError())
  {
    QXmlStreamReader::TokenType token = reader.readNext();
    if (token == QXmlStreamReader::StartDocument)
    {
      continue;
    }
    if (token == QXmlStreamReader::StartElement)
    {
      if (reader.name() == "state")
      {
        std::cout << "state: " << reader.attributes()[0].name().toString().toStdString() << std::endl; //reader.readElementText(QXmlStreamReader::IncludeChildElements).toStdString() << std::endl;
      }
      if (reader.name() == "transition")
        std::cout << "transition: " << std::endl;//reader.readElementText(QXmlStreamReader::IncludeChildElements).toStdString() << std::endl;
    }
  }

  if (reader.hasError())
    std::cerr << "Error while parsing: " << reader.errorString().toStdString() << std::endl;
}

ScxmlImporter::~ScxmlImporter()
{
}
