import QtQuick 2.0
import QtQuick.Controls 1.2

Item {
    id: item1
    x: 0
    y: 0
    width: 1280
    height: 720

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
