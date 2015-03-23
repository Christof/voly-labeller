#include "./scxml_importer.h"
#include <QXmlStreamReader>
#include <QFile>
#include <QStateMachine>
#include <QEvent>
#include <QKeySequence>
#include <QKeyEventTransition>
#include <iostream>
#include <cassert>

// from
// http://stackoverflow.com/questions/14034209/convert-string-representation-of-keycode-to-qtkey-or-any-int-and-back
uint toKey(QString const &str)
{
  QKeySequence seq(str);
  uint keyCode;

  // We should only working with a single key here
  if (seq.count() == 1)
    keyCode = seq[0];
  else
  {
    // Should be here only if a modifier key (e.g. Ctrl, Alt) is pressed.
    assert(seq.count() == 0);

    // Add a non-modifier key "A" to the picture because QKeySequence
    // seems to need that to acknowledge the modifier. We know that A has
    // a keyCode of 65 (or 0x41 in hex)
    seq = QKeySequence(str + "+A");
    assert(seq.count() == 1);
    assert(seq[0] > 65);
    keyCode = seq[0] - 65;
  }

  return keyCode;
}

ScxmlImporter::ScxmlImporter(QUrl url)
{
  QFile file(url.toLocalFile());
  std::cout << file.exists() << std::endl;
  file.open(QFile::OpenModeFlag::ReadOnly);

  stateMachine = std::shared_ptr<QStateMachine>(new QStateMachine());

  QXmlStreamReader reader(file.readAll());
  State *state;
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
        state =
            new State(reader.attributes().value("id").toString().toStdString());
        stateMachine->addState(state);

        std::cout << "state: " << state->name << std::endl;
        // reader.readElementText(QXmlStreamReader::IncludeChildElements).toStdString()
        // << std::endl;
      }
      if (reader.name() == "transition")
      {
        auto event = reader.attributes().value("event").toString();
        auto target = reader.attributes().value("target").toString();
        QString keyboardEvent = "KeyboardEvent";
        if (event.startsWith(keyboardEvent))
        {
          QEvent::Type eventType = QEvent::KeyPress;
          auto keyAsString = event.mid(event.lastIndexOf(".") + 1);
          auto keyCode = toKey(keyAsString);
          auto transition =
              new QKeyEventTransition(nullptr, eventType, keyCode);
          state->addTransition(transition);
          std::cout << keyAsString.toStdString() << ": " << keyCode << "|"
                    << std::to_string(Qt::Key_A) << std::endl;
        }
        std::cout << "transition: " << event.toStdString() << " | "
                  << target.toStdString() << std::endl;
      }
    }
  }

  if (reader.hasError())
    std::cerr << "Error while parsing: " << reader.errorString().toStdString()
              << std::endl;
}

ScxmlImporter::~ScxmlImporter()
{
}
