# PyQt5 imports

from PyQt5 import QtWidgets, QtCore, QtGui # Qt5
from OpenGL.GL import *
from OpenGL.GL.shaders import *
from OpenGL.GLU import *
import numpy


class GLPlotWidget(QtWidgets.QOpenGLWidget):
    # default window size
    width, height = 600, 600

    def __init__(self):
        self.program = None
        self.vertex = None
        self.fragment = None
        self.aVert = None
        self.auV = None
        self.utexY = None
        self.utexU = None
        self.utexV = None
        self.vertexBuffer = None
        self.uvShader = None
        self.y = None
        self.v = None
        self.u = None
        self.textures = None

    # vertex shader program
    vertexShader = """
        #version 300 es

        layout(location = 0) in vec3 vert;
        layout(location = 1) in vec2 uV;
        out vec2 UV;

        void main() {
          gl_Position = vec4(vert, 1.0);
          UV = uV;
        }
    """

    # fragment shader program
    # from https://gist.github.com/varphone/e21618adcdb687ce5c6cf54021593949
    fragmentShader = """
        #version 300 es
        precision mediump float;
        in vec2 UV;
        uniform sampler2D texY; // Y
        uniform sampler2D texU; // U
        uniform sampler2D texV; // V

        out vec4 colour;

        vec3 yuv2rgb(in vec3 yuv)
        {
            // YUV offset
            // const vec3 offset = vec3(-0.0625, -0.5, -0.5);
            const vec3 offset = vec3(-0.0625, -0.5, -0.5);
            // RGB coefficients
            const vec3 Rcoeff = vec3( 1.164, 0.000,  1.596);
            const vec3 Gcoeff = vec3( 1.164, -0.391, -0.813);
            const vec3 Bcoeff = vec3( 1.164, 2.018,  0.000);

            vec3 rgb;

            yuv = clamp(yuv, 0.0, 1.0);

            yuv += offset;

            rgb.r = dot(yuv, Rcoeff);
            rgb.g = dot(yuv, Gcoeff);
            rgb.b = dot(yuv, Bcoeff);
            return rgb;
        }

        vec3 get_yuv_from_texture(in vec2 tcoord)
        {
            vec3 yuv;
            yuv.x = texture(texY, tcoord).r;
            // Get the U and V values
            yuv.y = texture(texU, tcoord).r;
            yuv.z = texture(texV, tcoord).r;
            return yuv;

        }

        vec4 mytexture2D(in vec2 tcoord)
        {
            vec3 rgb, yuv;
            yuv = get_yuv_from_texture(tcoord);
            // Do the color transform
            rgb = yuv2rgb(yuv);
            return vec4(rgb, 1.0);
        }

        void main() {
          //colour = texture(texY, UV).rgba;
          colour = mytexture2D(UV);
        }
    """


    def get_image(self, filename=None):
        """ make texture based on image """
        from PIL import Image

        img = Image.open(filename)  # .jpg, .bmp, etc. also work
        img_data = numpy.array(list(img.getdata()), numpy.uint8)

        return (img, img_data)


    def get_image_yuv(self, filename=None):
        """ read image in luma chroma format
        identify hello2.tga     gives 400 x 300
        convert  hello2.tga hello2.yuv
        """
        f=open(filename,"b+r")
        st=f.read()
        f.close()

        ix=400
        iy=300

        n=ix*iy # assume 720p
        cn=0

        a=numpy.frombuffer(st,dtype=numpy.uint8)
        #y=a[cn:cn+n].reshape((720,1280))
        y=numpy.zeros((n,1))
        y[:,0]=a[cn:cn+n].reshape(n)
        cn+=n

        # 4:2:2
        u=numpy.zeros((n/4,1))
        u[:,0]=a[cn:cn+n/4].reshape(n/4)
        cn+=n/4

        v=numpy.zeros((n/4,1))
        v[:,0]=a[cn:cn+n/4].reshape(n/4)
        return(ix, iy, y, u, v)


    def initializeGL(self):
        # create shader program

        self.program = glCreateProgram()
        self.vertex = glCreateShader(GL_VERTEX_SHADER)
        self.fragment = glCreateShader(GL_FRAGMENT_SHADER)

        # Set shaders source
        glShaderSource(self.vertex, self.vertexShader)
        glShaderSource(self.fragment, self.fragmentShader)

        # Compile shaders
        glCompileShader(self.vertex)
        glCompileShader(self.fragment)

        # Attach shader objects to the program
        glAttachShader(self.program, self.vertex)
        glAttachShader(self.program, self.fragment)

        # Build program
        glLinkProgram(self.program)

        # Get rid of shaders (not needed anymore)
        glDetachShader(self.program, self.vertex)
        glDetachShader(self.program, self.fragment)

        # obtain uniforms and attributes
        self.aVert = glGetAttribLocation(self.program, "vert")
        self.aUV = glGetAttribLocation(self.program, "uV")
        self.utexY = glGetUniformLocation(self.program, "texY")
        self.utexU = glGetUniformLocation(self.program, "texU")
        self.utexV = glGetUniformLocation(self.program, "texV")

        # set  vertices
        Vertices = [
            -1.0,  1.0, 0.0,
            -1.0, -1.0, 0.0,
             1.0,  1.0, 0.0,
             1.0,  1.0, 0.0,
            -1.0, -1.0, 0.0,
             1.0, -1.0, 0.0]

        self.vertexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        vertexData = numpy.array(Vertices, numpy.float32)
        glBufferData(GL_ARRAY_BUFFER, 4 * len(vertexData), vertexData, GL_STATIC_DRAW)

        # set  UV
        UV = [
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0]

        self.uvBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.uvBuffer)
        uvData = numpy.array(UV, numpy.float32)
        glBufferData(GL_ARRAY_BUFFER, 4 * len(uvData), uvData, GL_STATIC_DRAW)


    def paintGL(self):
        """Paint the scene.
        """
        # use shader program
        glUseProgram(self.program)

        # set uniforms
        glUniform1i(self.utexY, 0)
        glUniform1i(self.utexU, 1)
        glUniform1i(self.utexV, 2)

        # enable attribute arrays
        glEnableVertexAttribArray(self.aVert)
        glEnableVertexAttribArray(self.aUV)

        # set vertex and UV buffers
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glVertexAttribPointer(self.aVert, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.uvBuffer)
        glVertexAttribPointer(self.aUV, 2, GL_FLOAT, GL_FALSE, 0, None)

        # read in yov figure, u and v have lower resolution compared to y
        (ix,iy,self.y,self.u,self.v)=self.get_image_yuv('hello2.yuv')

        self.textures = glGenTextures(3)
        glBindTexture(GL_TEXTURE_2D, self.textures[0])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, ix, iy, 0, GL_RED, GL_UNSIGNED_BYTE, self.y)

        glBindTexture(GL_TEXTURE_2D, self.textures[1])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, ix/2, iy/2, 0, GL_RED, GL_UNSIGNED_BYTE, self.u)

        glBindTexture(GL_TEXTURE_2D, self.textures[2])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, ix/2, iy/2, 0, GL_RED, GL_UNSIGNED_BYTE, self.v)

        # bind  textures
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, self.textures[0])
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, self.textures[1])
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, self.textures[2])

        # draw
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # disable attribute arrays
        glDisableVertexAttribArray(self.aVert)
        glDisableVertexAttribArray(self.aUV)


    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport.
        """
        # update the window size
        self.width, self.height = width, height
        # paint within the whole window
        glViewport(0, 0, width, height)
        # set orthographic projection (2D only)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # the window corner OpenGL coordinates are (-+1, -+1)
        glOrtho(-1, 1, 1, -1, -1, 1)


if __name__ == '__main__':
    # import numpy for generating random data points
    import sys
    import numpy as np
    import numpy.random as rdn

    # define a Qt window with an OpenGL widget inside it
    class TestWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()
            self.widget = GLPlotWidget()
            # put the window at the screen position (100, 100)
            self.setGeometry(100, 100, self.widget.width, self.widget.height)
            self.setCentralWidget(self.widget)
            self.show()

    # create the Qt App and window
    app = QtWidgets.QApplication(sys.argv)
    window = TestWindow()
    window.show()
    app.exec_()

