import QtQuick 2.1
import QtQuick.Controls 1.2
import QtQuick.Dialogs 1.0

Item {
  id: root
  x: 0
  y: 0
  width: 1280
  height: 720

  function toggleVisibility() {
      root.visible = !root.visible;
  }

  Item {
    id: menuWrapper
    anchors.fill: parent

    MenuBar {
      id: myTopMenu
      Menu {
        title: "File"
        MenuItem {
          text: "Open"
          shortcut: "Ctrl+o"
          onTriggered: addSceneNodesFromDialog.open();
        }
        MenuItem {
          text: "Import"
          shortcut: "Ctrl+i"
          onTriggered: importDialog.open();
        }
        MenuItem {
          text: "Save"
          shortcut: "Ctrl+s"
          onTriggered: saveSceneDialog.open();
        }
        MenuItem {
          text: "Reset scene"
          shortcut: "Ctrl+r"
          onTriggered: nodes.clear();
        }
        MenuItem {
          text: "Hide user interface"
          shortcut: "F1"
          onTriggered: toggleVisibility();
        }
        MenuItem {
          text: "Toggle bounding volumes"
          shortcut: "F2"
          onTriggered: nodes.toggleBoundingVolumes();
        }
        MenuItem {
          text: "Display state machine state"
          shortcut: "l"
          onTriggered: window.printCurrentState();
        }
        MenuItem {
          text: "Exit"
          shortcut: "Esc"
          onTriggered: Qt.quit();
        }
      }
    }

    FileDialog {
        id: addSceneNodesFromDialog
        title: "Please choose a scene file"
        nameFilters: [ "Xml files (*.xml)" ]
        onAccepted: {
            nodes.addSceneNodesFrom(fileUrl);
        }
    }

    FileDialog {
        id: importDialog
        title: "Please choose a file to import"
        nameFilters: [ "Collada files (*.dae)" ]
        onAccepted: {
            nodes.importFrom(fileUrl);
        }
    }

    FileDialog {
        id: saveSceneDialog
        selectExisting: false
        title: "Please choose a scene file"
        nameFilters: [ "Xml files (*.xml)" ]
        onAccepted: {
            nodes.saveSceneTo(fileUrl);
        }
    }

    states: State {
      name: "hasMenuBar"
      when: myTopMenu && !myTopMenu.__isNative

      ParentChange {
        target: myTopMenu.__contentItem
        parent: root
      }

      PropertyChanges {
        target: myTopMenu.__contentItem
        x: 0
        y: 0
        width: menuWrapper.width
      }
    }

    Component {
      id: forceDelegate
      Item {
        width: 400; height: 30
        Row {
          CheckBox { checked: enabled }
          Text { text: name }
          TextInput { text: weight }
        }
      }
    }

    ListView {
        x: 10; y: 30
        width: 400; height: 250
        model: labeller
        delegate: forceDelegate
    }
  }
}
