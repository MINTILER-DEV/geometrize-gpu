INCLUDEPATH += $$PWD

HEADERS += $$files($$PWD/geometrize/*.h, true)
SOURCES += $$files($$PWD/geometrize/*.cpp, true)
DISTFILES += $$files($$PWD/../shaders/*.comp, true)

*-g++* {
    QMAKE_CXXFLAGS += -pthread
    LIBS += -pthread
}

contains(DEFINES, GEOMETRIZE_ENABLE_OPENGL_COMPUTE) {
    win32 {
        LIBS += -lopengl32 -lglew32
    }
    unix:!macx {
        LIBS += -lGL -lGLEW
    }
    macx {
        LIBS += -framework OpenGL
    }
}
