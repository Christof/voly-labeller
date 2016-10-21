#include "./application.h"
#include <QQmlContext>
#include <QStateMachine>
#include <QDebug>
#include <vector>
#include "./window.h"
#include "./scene.h"
#include "./scene_controller.h"
#include "./nodes.h"
#include "./nodes_controller.h"
#include "./label_node.h"
#include "./input/invoke_manager.h"
#include "./input/signal_manager.h"
#include "./input/scxml_importer.h"
#include "./mouse_shape_controller.h"
#include "./labeller_model.h"
#include "./placement_labeller_model.h"
#include "./labels_model.h"
#include "./camera_positions_model.h"
#include "./labelling/labels.h"
#include "./labelling_coordinator.h"
#include "./labelling_controller.h"
#include "./picking_controller.h"
#include "./forces_visualizer_node.h"
#include "./camera_node.h"
#include "./default_scene_creator.h"
#include "./video_recorder.h"
#include "./video_recorder_controller.h"
#include "./texture_mapper_manager.h"
#include "./texture_mapper_manager_controller.h"
#include "./utils/memory.h"
#include "./utils/path_helper.h"
#include "./recording_automation.h"
#include "./recording_automation_controller.h"

const int LAYER_COUNT = 4;

Application::Application(int &argc, char **argv) : application(argc, argv)
{
  application.setApplicationName("voly-labeller");
  application.setApplicationVersion("0.0.1");
  application.setOrganizationName("LBI");

  setupCommandLineParser();
  parser.process(application);
  invokeManager = std::make_shared<InvokeManager>();

  nodes = std::make_shared<Nodes>();
  nodesController = std::make_shared<NodesController>(nodes);

  labels = std::make_shared<Labels>();
  forcesLabeller = std::make_shared<Forces::Labeller>(labels);

  int layerCount = parseLayerCount();

  auto labellingCoordinator = std::make_shared<LabellingCoordinator>(
      layerCount, forcesLabeller, labels, nodes);

  bool synchronousCapturing = parser.isSet("offline");
  const float offlineFPS = 24;
  videoRecorder =
      std::make_shared<VideoRecorder>(synchronousCapturing, offlineFPS);
  videoRecorderController =
      std::make_unique<VideoRecorderController>(videoRecorder);
  recordingAutomation = std::make_shared<RecordingAutomation>(
      labellingCoordinator, nodes, videoRecorder);
  if (parser.isSet("screenshot"))
    recordingAutomation->takeScreenshotOfPositionAndExit(
        parser.value("screenshot").toStdString());

  recordingAutomationController =
      std::make_unique<RecordingAutomationController>(recordingAutomation);

  const int postProcessingTextureSize = 512;
  textureMapperManager =
      std::make_shared<TextureMapperManager>(postProcessingTextureSize);
  textureMapperManagerController =
      std::make_unique<TextureMapperManagerController>(textureMapperManager);
  scene = std::make_shared<Scene>(layerCount, invokeManager, nodes, labels,
                                  labellingCoordinator, textureMapperManager,
                                  recordingAutomation);

  float offlineRenderingFrameTime =
      parser.isSet("offline") ? 1.0f / offlineFPS : 0.0f;
  window =
      std::make_unique<Window>(scene, videoRecorder, offlineRenderingFrameTime);

  sceneController = std::make_unique<SceneController>(scene);
  labellerModel = std::make_unique<LabellerModel>(forcesLabeller);
  placementLabellerModel =
      std::make_unique<PlacementLabellerModel>(labellingCoordinator);
  cameraPositionsModel = std::make_unique<CameraPositionsModel>(nodes);
  mouseShapeController = std::make_unique<MouseShapeController>();
  pickingController = std::make_shared<PickingController>(scene);
  labelsModel = std::make_unique<LabelsModel>(labels, pickingController);
  labellingController =
      std::make_unique<LabellingController>(labellingCoordinator);
}

Application::~Application()
{
}

int Application::execute()
{
  qInfo() << "Application start";

  auto unsubscribeLabelChanges = labels->subscribe(
      std::bind(&Application::onLabelChangedUpdateLabelNodes, this,
                std::placeholders::_1, std::placeholders::_2));

  setupWindow();

  connect(labellerModel.get(), &LabellerModel::isVisibleChanged, this,
          &Application::onFocesLabellerModelIsVisibleChanged);

  window->setSource(QUrl("qrc:ui.qml"));

  forcesLabeller->resize(window->size().width(), window->size().height());

  nodes->setOnNodeAdded(
      [this](std::shared_ptr<Node> node) { this->onNodeAdded(node); });

  nodes->getCameraNode()->setOnCameraPositionsChanged(
      [this](std::vector<CameraPosition> cameraPositions) {
        this->cameraPositionsModel->update(cameraPositions);
      });

  if (parser.positionalArguments().size())
  {
    auto absolutePath = QDir(parser.positionalArguments()[0]).absolutePath();
    auto filename = absolutePathToProjectRelativePath(absolutePath);
    qInfo() << "import scene:" << filename;
    nodesController->addSceneNodesFrom(filename);
  }
  else
  {
    DefaultSceneCreator sceneCreator(nodes, labels);
    sceneCreator.create();
  }

  createAndStartStateMachine();

  window->show();

  auto resultCode = application.exec();

  unsubscribeLabelChanges();

  return resultCode;
}

void Application::setupCommandLineParser()
{
  QGuiApplication::setApplicationName("voly-labeller");
  QGuiApplication::setApplicationVersion("0.1");

  parser.setApplicationDescription(
      "Multiple labelling implementations for volume rendered medical data");
  parser.addHelpOption();
  parser.addVersionOption();
  parser.addPositionalArgument(
      "scene", QCoreApplication::translate("main", "Scene file to load."));
  QCommandLineOption offlineRenderingOption("offline",
                                            "Enables offline rendering");
  parser.addOption(offlineRenderingOption);

  QCommandLineOption layersOption("layers", "Number of layers. Default is 4",
                                  "layerCount", "4");
  parser.addOption(layersOption);

  QCommandLineOption screenshotOption(
      QStringList() << "s"
                    << "screenshot",
      "Takes a screenshot of the given camera position", "Camera Position");
  parser.addOption(screenshotOption);
}

void Application::setupWindow()
{
  window->setResizeMode(QQuickView::SizeRootObjectToView);
  auto context = window->rootContext();
  auto assetsPath =
      QUrl::fromLocalFile(absolutePathOfProjectRelativePath(QString("assets")));
  context->setContextProperty("assetsPath", assetsPath);
  auto projectRootPath =
      QUrl::fromLocalFile(absolutePathOfProjectRelativePath(QString(".")));
  context->setContextProperty("projectRootPath", projectRootPath);

  context->setContextProperty("window", window.get());
  context->setContextProperty("nodes", nodesController.get());
  context->setContextProperty("bufferTextures",
                              textureMapperManagerController.get());
  context->setContextProperty("scene", sceneController.get());
  context->setContextProperty("labeller", labellerModel.get());
  context->setContextProperty("placement", placementLabellerModel.get());
  context->setContextProperty("cameraPositions", cameraPositionsModel.get());
  context->setContextProperty("labels", labelsModel.get());
  context->setContextProperty("labelling", labellingController.get());
  context->setContextProperty("videoRecorder", videoRecorderController.get());
  context->setContextProperty("automation",
                              recordingAutomationController.get());
}

void Application::createAndStartStateMachine()
{
  auto signalManager = std::shared_ptr<SignalManager>(new SignalManager());
  ScxmlImporter importer(QUrl::fromLocalFile("config/states.xml"),
                         invokeManager, signalManager);

  invokeManager->addHandler(window.get());
  invokeManager->addHandler("mouseShape", mouseShapeController.get());
  invokeManager->addHandler("picking", pickingController.get());
  invokeManager->addHandler("scene", sceneController.get());
  signalManager->addSender("KeyboardEventSender", window.get());
  signalManager->addSender("window", window.get());
  signalManager->addSender("labels", labelsModel.get());

  stateMachine = importer.import();

  // just for printCurrentState slot for debugging
  window->stateMachine = stateMachine;

  stateMachine->start();
}

void Application::onNodeAdded(std::shared_ptr<Node> node)
{
  std::shared_ptr<LabelNode> labelNode =
      std::dynamic_pointer_cast<LabelNode>(node);
  if (labelNode.get())
  {
    labelNode->anchorSize = nodesController->getAnchorSize();
    labels->add(labelNode->label);
  }

  std::shared_ptr<CameraNode> cameraNode =
      std::dynamic_pointer_cast<CameraNode>(node);
  if (cameraNode.get())
  {
    cameraNode->setOnCameraPositionsChanged(
        [this](std::vector<CameraPosition> cameraPositions) {
          this->cameraPositionsModel->update(cameraPositions);
        });
  }
}

void Application::onLabelChangedUpdateLabelNodes(Labels::Action action,
                                                 const Label &label)
{
  auto labelNodes = nodes->getLabelNodes();
  auto labelNode = std::find_if(labelNodes.begin(), labelNodes.end(),
                                [label](std::shared_ptr<LabelNode> labelNode) {
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

void Application::onFocesLabellerModelIsVisibleChanged()
{
  if (labellerModel->getIsVisible())
  {
    nodes->addForcesVisualizerNode(
        std::make_shared<ForcesVisualizerNode>(forcesLabeller));
  }
  else
  {
    nodes->removeForcesVisualizerNode();
  }
}

int Application::parseLayerCount()
{
  int layerCount = 4;
  if (parser.isSet("layers"))
  {
    bool gotLayerCount = true;
    layerCount = parser.value("layers").toInt(&gotLayerCount);
    if (!gotLayerCount)
    {
      layerCount = 4;
      qWarning() << "Problem parsing layer count from" << parser.value("layers");
    }
  }

  return layerCount;
}

