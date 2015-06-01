#include <QtGui/QGuiApplication>
#include <QQmlContext>
#include <QStateMachine>
#include <QDebug>
#include <memory>
#include "./window.h"
#include "./scene.h"
#include "./nodes.h"
#include "./input/invoke_manager.h"
#include "./input/signal_manager.h"
#include "./input/scxml_importer.h"
#include "./mouse_shape_controller.h"
#include "./labeller_model.h"
#include "./labels_model.h"
#include "./labelling/labels.h"
#include "./picking_controller.h"
#include "./forces_visualizer_node.h"

int main(int argc, char **argv)
{
  qputenv("QT_MESSAGE_PATTERN",
          QString("%{time [yyyy'-'MM'-'dd' 'hh':'mm':'ss]} - %{threadid} "
                  "%{if-category}%{category}: %{endif}%{message}").toUtf8());
  if (qgetenv("QT_LOGGING_CONF").size() == 0)
    qputenv("QT_LOGGING_CONF", "../config/logging.ini");

  QGuiApplication application(argc, argv);

  auto invokeManager = std::shared_ptr<InvokeManager>(new InvokeManager());
  auto nodes = std::make_shared<Nodes>();
  auto labels = std::make_shared<Labels>();
  auto labeller = std::make_shared<Forces::Labeller>(labels);
  auto forcesVisualizerNode = std::make_shared<ForcesVisualizerNode>(labeller);
  nodes->addNode(forcesVisualizerNode);
  auto scene = std::make_shared<Scene>(invokeManager, nodes, labels, labeller);

  Window window(scene);
  window.setResizeMode(QQuickView::SizeRootObjectToView);
  window.rootContext()->setContextProperty("window", &window);
  window.rootContext()->setContextProperty("nodes", nodes.get());

  MouseShapeController mouseShapeController;
  PickingController pickingController(scene);

  LabellerModel labellerModel(labeller);
  labellerModel.connect(&labellerModel, &LabellerModel::isVisibleChanged,
                        [&labellerModel, &nodes, &forcesVisualizerNode]()
                        {
    if (labellerModel.getIsVisible())
      nodes->addNode(forcesVisualizerNode);
    else
      nodes->removeNode(forcesVisualizerNode);
  });
  window.rootContext()->setContextProperty("labeller", &labellerModel);

  LabelsModel labelsModel(nodes, pickingController);
  window.rootContext()->setContextProperty("labels", &labelsModel);
  window.setSource(QUrl("qrc:ui.qml"));

  auto signalManager = std::shared_ptr<SignalManager>(new SignalManager());
  ScxmlImporter importer(QUrl::fromLocalFile("config/states.xml"),
                         invokeManager, signalManager);

  invokeManager->addHandler(&window);
  invokeManager->addHandler("mouseShape", &mouseShapeController);
  invokeManager->addHandler("picking", &pickingController);
  signalManager->addSender("KeyboardEventSender", &window);
  signalManager->addSender("labels", &labelsModel);

  auto stateMachine = importer.import();

  // just for printCurrentState slot for debugging
  window.stateMachine = stateMachine;

  stateMachine->start();

  window.show();

  return application.exec();
}
