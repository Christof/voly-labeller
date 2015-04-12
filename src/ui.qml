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
          onTriggered: fileDialog.open();
        }
        MenuItem {
          text: "Hide user interface"
          shortcut: "F1"
          onTriggered: toggleVisibility();
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
        id: fileDialog
        title: "Please choose a scene file"
        nameFilters: [ "Xml files (*.xml)" ]
        onAccepted: {
            window.openScene(fileDialog.fileUrl);
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
  }

  Rectangle {
    id: rectangle1
    x: 790
    y: 57
    width: 200
    height: 200
    radius: 3
    z: 1
    opacity: 1
    border.width: 4
    gradient: Gradient {
      GradientStop {
        position: 0
        color: "#ffffff"
      }

      GradientStop {
        position: 1
        color: "#146d78"
      }
    }
  }

  Slider {
    id: sliderHorizontal1
    x: 8
    y: 688
    width: 1264
    height: 24
    visible: true
    clip: false
    opacity: 1
  }
}
