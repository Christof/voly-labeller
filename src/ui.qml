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
          CheckBox {
            checked: enabled
            /*
            onClicked: enabled = checked
            Component.onCompleted: checked = enabled
            Connections {
              target: labeller
              onDataChanged: {
                enabled = checked
              }
            }
            */
          }
          Text {
            text: name
            width: 200
            focus: true
          }
          TextEdit { text: weight }
        }
      }
    }

    Component {
      id: checkBoxDelegate

      Item {
        CheckBox {
          anchors.fill: parent
          checked: styleData.value
          onCheckedChanged: {
            console.log("changed");
            labeller.changeEnabled(styleData.row, checked);
          }
        }
      }
    }

    Component {
      id: textDelegate

      Item {
        TextInput {
          anchors.fill: parent
          text: styleData.value
          onTextChanged: {
            labeller.changeWeight(styleData.row, text);
          }
        }
      }
    }

    TableView {
      TableViewColumn {
        role: "enabled"
        title: "Enabled"
        width: 100
        delegate: checkBoxDelegate
      }
      TableViewColumn {
        role: "name"
        title: "Name"
        width: 200
      }
      TableViewColumn {
        role: "weight"
        title: "Weight"
        delegate: textDelegate
      }
      x: 10; y: 30
      width: 400
      model: labeller
      focus: true
      clip: true
    }
  }
}
