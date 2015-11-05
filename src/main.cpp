#include <QtGui/QGuiApplication>
#include <QQmlContext>
#include <QStateMachine>
#include <QDebug>
#include <cuda_runtime.h>
#include <memory>
#include "./window.h"
#include "./scene.h"
#include "./nodes.h"
#include "./label_node.h"
#include "./input/invoke_manager.h"
#include "./input/signal_manager.h"
#include "./input/scxml_importer.h"
#include "./mouse_shape_controller.h"
#include "./labeller_model.h"
#include "./labels_model.h"
#include "./labelling/labels.h"
#include "./picking_controller.h"
#include "./forces_visualizer_node.h"
#include "./placement/labeller.h"
#include "./default_scene_creator.h"
#include "./texture_mapper_manager.h"
#include "./utils/cuda_helper.h"

void onLabelChangedUpdateLabelNodes(std::shared_ptr<Nodes> nodes,
                                    Labels::Action action, const Label &label)
{
  auto labelNodes = nodes->getLabelNodes();
  auto labelNode = std::find_if(labelNodes.begin(), labelNodes.end(),
                                [label](std::shared_ptr<LabelNode> labelNode)
                                {
    return labelNode->label.id == label.id;
  });

  if (labelNode == labelNodes.end())
  {
    nodes->addNode(std::make_shared<LabelNode>(label));
  }
  else if (action == Labels::Action::Delete)
  {
    nodes->removeNode(*labelNode);
  }
  else
  {
    (*labelNode)->label = label;
  }
};

/**
 * \brief Setup logging
 *
 * [Documentation for the
 * pattern](http://doc.qt.io/qt-5/qtglobal.html#qSetMessagePattern)
 *
 * [Documentation for console formatting]
 * (https://en.wikipedia.org/wiki/ANSI_escape_code#Colors)
 */
void setupLogging()
{
  qputenv("QT_MESSAGE_PATTERN",
          QString("%{time [yyyy'-'MM'-'dd' "
                  "'hh':'mm':'ss]} "
                  "%{if-fatal}\033[31;1m%{endif}"
                  "%{if-critical}\033[31m%{endif}"
                  "%{if-warning}\033[33m%{endif}"
                  "%{if-info}\033[34m%{endif}"
                  "- %{threadid} "
                  "%{if-category}%{category}: %{endif}%{message}"
                  "%{if-warning}\n\t%{file}:%{line}\n\t%{backtrace depth=3 "
                  "separator=\"\n\t\"}%{endif}"
                  "%{if-critical}\n\t%{file}:%{line}\n\t%{backtrace depth=3 "
                  "separator=\"\n\t\"}%{endif}\033[0m").toUtf8());
  if (qgetenv("QT_LOGGING_CONF").size() == 0)
    qputenv("QT_LOGGING_CONF", "../config/logging.ini");
}

void setupCuda()
{
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0)
  {
    qCritical() << "No cuda device found!";
    exit(EXIT_FAILURE);
  }

  cudaDeviceProp prop;
  int device;
  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 3;
  prop.minor = 0;
  HANDLE_ERROR(cudaChooseDevice(&device, &prop));
  HANDLE_ERROR(cudaGLSetGLDevice(device));
}

int main(int argc, char **argv)
{
  setupLogging();

  setupCuda();

  QGuiApplication application(argc, argv);

  qInfo() << "Application start";

  auto invokeManager = std::shared_ptr<InvokeManager>(new InvokeManager());
  auto nodes = std::make_shared<Nodes>();
  auto labels = std::make_shared<Labels>();
  auto forcesLabeller = std::make_shared<Forces::Labeller>(labels);
  auto placementLabeller = std::make_shared<Placement::Labeller>(labels);

  const int postProcessingTextureSize = 512;
  auto textureMapperManager =
      std::make_shared<TextureMapperManager>(postProcessingTextureSize);

  auto scene =
      std::make_shared<Scene>(invokeManager, nodes, labels, forcesLabeller,
                              placementLabeller, textureMapperManager);

  auto unsubscribeLabelChanges = labels->subscribe(
      std::bind(&onLabelChangedUpdateLabelNodes, nodes, std::placeholders::_1,
                std::placeholders::_2));

  std::unique_ptr<Window> window = std::unique_ptr<Window>(new Window(scene));
  window->setResizeMode(QQuickView::SizeRootObjectToView);
  window->rootContext()->setContextProperty("window", window.get());
  window->rootContext()->setContextProperty("nodes", nodes.get());

  MouseShapeController mouseShapeController;
  PickingController pickingController(scene);

  LabellerModel labellerModel(forcesLabeller);
  labellerModel.connect(&labellerModel, &LabellerModel::isVisibleChanged,
                        [&labellerModel, &nodes, &forcesLabeller]()
                        {
    if (labellerModel.getIsVisible())
      nodes->addForcesVisualizerNode(
          std::make_shared<ForcesVisualizerNode>(forcesLabeller));
    else
      nodes->removeForcesVisualizerNode();
  });
  window->rootContext()->setContextProperty("labeller", &labellerModel);

  LabelsModel labelsModel(labels, pickingController);
  window->rootContext()->setContextProperty("labels", &labelsModel);
  window->setSource(QUrl("qrc:ui.qml"));

  forcesLabeller->resize(window->size().width(), window->size().height());
  DefaultSceneCreator sceneCreator(nodes, labels);
  sceneCreator.create();

  auto signalManager = std::shared_ptr<SignalManager>(new SignalManager());
  ScxmlImporter importer(QUrl::fromLocalFile("config/states.xml"),
                         invokeManager, signalManager);

  invokeManager->addHandler(window.get());
  invokeManager->addHandler("mouseShape", &mouseShapeController);
  invokeManager->addHandler("picking", &pickingController);
  signalManager->addSender("KeyboardEventSender", window.get());
  signalManager->addSender("labels", &labelsModel);

  auto stateMachine = importer.import();

  // just for printCurrentState slot for debugging
  window->stateMachine = stateMachine;

  stateMachine->start();

  window->show();

  auto resultCode = application.exec();

  unsubscribeLabelChanges();

  return resultCode;
}

