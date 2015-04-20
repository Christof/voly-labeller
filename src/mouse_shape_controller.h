#ifndef SRC_MOUSE_SHAPE_CONTROLLER_H_

#define SRC_MOUSE_SHAPE_CONTROLLER_H_

#include <QObject>
#include <QtQuick/QQuickView>

/**
 * \brief
 *
 *
 */
class MouseShapeController : public QObject
{
  Q_OBJECT
 public:
  MouseShapeController(QQuickView &view) : view(view)
  {
  }

 public slots:
  void startDragging()
  {
    auto cursor = view.cursor();
    cursor.setShape(Qt::CursorShape::ClosedHandCursor);
    view.setCursor(cursor);
  }

  void endDragging()
  {
    auto cursor = view.cursor();
    cursor.setShape(Qt::CursorShape::ArrowCursor);
    view.setCursor(cursor);
  }

 private:
  QQuickView &view;
};

#endif  // SRC_MOUSE_SHAPE_CONTROLLER_H_
