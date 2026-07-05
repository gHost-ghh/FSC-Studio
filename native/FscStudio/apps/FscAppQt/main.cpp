#include <QApplication>
#include <QLabel>

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QLabel label("FSC Studio Native Qt: algorithm parity build first");
    label.resize(720, 160);
    label.show();
    return app.exec();
}
