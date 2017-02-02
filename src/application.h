#ifndef SRC_APPLICATION_H_

#define SRC_APPLICATION_H_

#include <QtGui/QGuiApplication>
#include <QCommandLineParser>
#include <QObject>
#include "./forces/labeller.h"

class InvokeManager;
class Node;
class Nodes;
class NodesController;
class Labels;
class TextureMapperManager;
class TextureMapperManagerController;

class Scene;
class Window;
class SceneController;
class LabellerModel;
class PlacementLabellerModel;
class CameraPositionsModel;
class LabellingController;
class MouseShapeController;
class PickingController;
class LabelsModel;
class QStateMachine;
class VideoRecorder;
class VideoRecorderController;
class RecordingAutomation;
class RecordingAutomationController;

/**
 * \brief Class for the whole application
 *
 * It holds pointers to the main objects and handles the command line
 * arguments.
 *
 * The application is started by calling #execute.
 */
class Application : public QObject
{
  Q_OBJECT;

 public:
  Application(int &argc, char **argv);
  virtual ~Application();

  int execute();

 private:
  QGuiApplication application;
  QCommandLineParser parser;
  std::shared_ptr<InvokeManager> invokeManager;
  std::shared_ptr<Nodes> nodes;
  std::shared_ptr<NodesController> nodesController;
  std::shared_ptr<Labels> labels;
  std::shared_ptr<Forces::Labeller> forcesLabeller;
  std::unique_ptr<LabellingController> labellingController;
  std::shared_ptr<TextureMapperManager> textureMapperManager;
  std::shared_ptr<TextureMapperManagerController>
      textureMapperManagerController;
  std::shared_ptr<Scene> scene;
  std::shared_ptr<VideoRecorder> videoRecorder;
  std::unique_ptr<VideoRecorderController> videoRecorderController;
  std::unique_ptr<Window> window;
  std::unique_ptr<SceneController> sceneController;
  std::shared_ptr<RecordingAutomation> recordingAutomation;
  std::unique_ptr<RecordingAutomationController> recordingAutomationController;
  std::unique_ptr<LabellerModel> labellerModel;
  std::unique_ptr<PlacementLabellerModel> placementLabellerModel;
  std::unique_ptr<CameraPositionsModel> cameraPositionsModel;
  std::unique_ptr<MouseShapeController> mouseShapeController;
  std::shared_ptr<PickingController> pickingController;
  std::unique_ptr<LabelsModel> labelsModel;
  std::shared_ptr<QStateMachine> stateMachine;

  void setupCommandLineParser();
  int parseLayerCount();
  void setupWindow();
  void createAndStartStateMachine();

  void onNodeAdded(std::shared_ptr<Node> node);
  void onLabelChangedUpdateLabelNodes(Labels::Action action,
                                      const Label &label);
  void onFocesLabellerModelIsVisibleChanged();
  void handleLabelScaling();

 private slots:
   void onInitializationDone();
};

#endif  // SRC_APPLICATION_H_
