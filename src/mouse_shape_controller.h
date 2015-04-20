#ifndef SRC_MOUSE_SHAPE_CONTROLLER_H_

#define SRC_MOUSE_SHAPE_CONTROLLER_H_

#include <QObject>
#include <QtQuick/QQuickView>

/**
 * \brief Changes the shape of the mouse cursor depending on the state
 *
 * The state is changed by the state machine using the provided slots.
 */
class MouseShapeController : public QObject
{
  Q_OBJECT
 public:
  explicit MouseShapeController(QQuickView &view) : view(view)
  {
  }

 public slots:
  void startDragging()
  {
    setShape(Qt::CursorShape::ClosedHandCursor);
  }

  void reset()
  {
    setShape(Qt::CursorShape::ArrowCursor);
  }

  void startZoom()
  {
    setShape(Qt::CursorShape::SizeVerCursor);
  }

 private:
  void setShape(Qt::CursorShape shape)
  {
    auto cursor = view.cursor();
    cursor.setShape(shape);
    view.setCursor(cursor);
  }

  QQuickView &view;
};

#endif  // SRC_MOUSE_SHAPE_CONTROLLER_H_
