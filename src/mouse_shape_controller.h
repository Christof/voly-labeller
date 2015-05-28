#ifndef SRC_MOUSE_SHAPE_CONTROLLER_H_

#define SRC_MOUSE_SHAPE_CONTROLLER_H_

#include <QObject>
#include <QApplication>
#include <QtQuick/QQuickView>

/**
 * \brief Changes the shape of the mouse cursor depending on the state
 *
 * The state is changed by the state machine using the provided slots.
 */
class MouseShapeController : public QObject
{
  Q_OBJECT
 public slots:
  void startDragging()
  {
    setShape(Qt::CursorShape::ClosedHandCursor);
  }

  void reset()
  {
    QApplication::restoreOverrideCursor();
  }

  void startZoom()
  {
    setShape(Qt::CursorShape::SizeVerCursor);
  }

  void startMove()
  {
    setShape(Qt::CursorShape::SizeAllCursor);
  }

  void startPicking()
  {
    setShape(Qt::CursorShape::CrossCursor);
  }

 private:
  void setShape(Qt::CursorShape shape)
  {
    QApplication::setOverrideCursor(shape);
  }
};

#endif  // SRC_MOUSE_SHAPE_CONTROLLER_H_
