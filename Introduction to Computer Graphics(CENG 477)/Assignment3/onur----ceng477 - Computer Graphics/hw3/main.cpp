#include "helper.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/vec3.hpp"
#include "glm/gtc/type_ptr.hpp"
#include <vector>

static GLFWwindow *window = nullptr;

// Shaders
GLuint idProgramShader;
GLuint idFragmentShader;
GLuint idVertexShader;
GLuint idJpegHeightMap;
GLuint idJpegTexture;
GLuint idMVPMatrix;

// Parameters
GLuint idHeightFactor;
GLuint idCameraPosition;
GLuint idWidthTexture;
GLuint idHeightTexture;

int widthTexture, heightTexture;
// initially
float heightFactor;
float pitchAngle;
float yawAngle = 90.0f;

// TODO ?
float speed = 0; // moving speed of the camera, initially zero

bool isFullscreen = false;
int oldPosX, oldPosY;

/* Data and their buffers */
// Vertices
GLfloat *g_vertex_buffer_data; // old triangle vertices
GLuint vertexbuffer;

// Our ModelViewProjection : multiplication of our 3 matrices
glm::mat4 *mvp;

/* Camera params */
glm::vec4 *camera_pos;
glm::vec3 *camera_gaze;
glm::vec3 *camera_up;

// Vertex array id -- for texture
GLuint VertexArrayId;

// Vertex array id -- for heightmap
GLuint VertexArrayIdHeightMap;

static void errorCallback(int error, const char * description) {
    fprintf(stderr, "Error: %s with error no:%d\n", description, error);
}

// Call this in while loop, I guess
void setMVP() {
    glm::vec3 camera_pos_vec3;
    camera_pos_vec3.x = camera_pos->x;
    camera_pos_vec3.y = camera_pos->y;
    camera_pos_vec3.z = camera_pos->z;

    // if we have speed, then we move the eye position in gaze direction
    camera_pos_vec3 += speed * (*camera_gaze);

    // Projection matrix
    glm::mat4 Projection = glm::perspective(45.0f, 1.0f, 0.1f, 1000.0f);
    // Camera matrix
    glm::mat4 View = glm::lookAt(camera_pos_vec3, (*camera_gaze) + camera_pos_vec3, *camera_up);
    // Model matrix : an identity matrix (model will be at the origin)
    glm::mat4 Model = glm::mat4(1.0f);
    // Our ModelViewProjection : multiplication of our 3 matrices
    *mvp = Projection * View * Model;

    camera_pos->x = camera_pos_vec3.x;
    camera_pos->y = camera_pos_vec3.y;
    camera_pos->z = camera_pos_vec3.z;
    //cout << camera_pos->x << "," << camera_pos->y << "," << camera_pos->z << endl;
    glUniformMatrix4fv(idMVPMatrix, 1, GL_FALSE, glm::value_ptr(*mvp));
    glUniform4fv(idCameraPosition, 1, glm::value_ptr(*camera_pos));    
}

// Keyboard functions, they are called in keyboard() function which is the key listener
void increaseHeightFactor(){
    heightFactor += 0.5;
    // get heightfactor parameter from shader and update it 
    glUniform1f(idHeightFactor, heightFactor);
}
void decreaseHeightFactor(){
    heightFactor -= 0.5;
    // get heightfactor parameter from shader and update it 
    glUniform1f(idHeightFactor, heightFactor);
}

void updateGaze(){
    camera_gaze->x = cos(glm::radians(pitchAngle)) * cos(glm::radians(yawAngle));
    camera_gaze->y = sin(glm::radians(pitchAngle));
    camera_gaze->z = cos(glm::radians(pitchAngle)) * sin(glm::radians(yawAngle));
}

void pitchUp(){
    pitchAngle += 0.6;
    updateGaze();
}
void pitchDown(){
    pitchAngle -= 0.6;
    updateGaze();
}
// TODO: yaw might need to be reversed
void yawLeft(){
    yawAngle -= 0.6;
    updateGaze();
}
void yawRight(){
    yawAngle += 0.6;
    updateGaze();
}

// TODO: hata olabilir bunlarda !
void increaseSpeed(){
    speed++;
}
void decreaseSpeed(){
    speed--;
}

void fullscreenToggle(){
    if (isFullscreen){
        // get back to windowed, set window monitor to null and set viewport back to default
        glfwSetWindowMonitor(window, NULL, oldPosX, oldPosY, 1000, 1000, GL_DONT_CARE);
        glViewport(0, 0, 1000, 1000);
        isFullscreen = false;
    }
    else{
        // get to fullscreen mode, get the primary monitor, set window monitor to this and set viewport accordingly
        glfwGetWindowPos(window, &oldPosX, &oldPosY);
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* videoMode = glfwGetVideoMode(monitor);
        glfwSetWindowMonitor (window, monitor, 0, 0, videoMode->width, videoMode->height, GL_DONT_CARE);
        glViewport(0, 0, videoMode->width, videoMode->height);
        isFullscreen = true;
    }
}

// Key event function which listens all key events
void keyboard(GLFWwindow *window, int key, int scancode, int action, int mods) {
    // height factor
    if (key == GLFW_KEY_O && (action == GLFW_PRESS || action == GLFW_REPEAT)) increaseHeightFactor();
    else if (key == GLFW_KEY_L && (action == GLFW_PRESS || action == GLFW_REPEAT)) decreaseHeightFactor();
    // pitch
    else if (key == GLFW_KEY_W && (action == GLFW_PRESS || action == GLFW_REPEAT)) pitchUp();
    else if (key == GLFW_KEY_S && (action == GLFW_PRESS || action == GLFW_REPEAT)) pitchDown();
    // yaw
    else if (key == GLFW_KEY_A && (action == GLFW_PRESS || action == GLFW_REPEAT)) yawLeft();
    else if (key == GLFW_KEY_D && (action == GLFW_PRESS || action == GLFW_REPEAT)) yawRight();
    // speed
    else if (key == GLFW_KEY_U && (action == GLFW_PRESS || action == GLFW_REPEAT)) increaseSpeed();
    else if (key == GLFW_KEY_J && (action == GLFW_PRESS || action == GLFW_REPEAT)) decreaseSpeed();
    // fullscreen
    else if (key == GLFW_KEY_F && action == GLFW_PRESS) fullscreenToggle();
}
 
// Window resize callback
void resize(GLFWwindow* window, int width, int height) {
    // TODO: implement
    //glfwGetWindowSize(win, &width, &height);
    glViewport(0, 0, width, height );
}

// XXX: Call this before windows is created and before any other OpenGL call
// function that initiates the vertex array in user domain
void initVertexArray(int width, int height) {
    glGenVertexArrays(1, &VertexArrayId);
    glBindVertexArray(VertexArrayId);
}

// XXX: Call this before windows is created and before any other OpenGL call
// function that initiates the vertex array in user domain
void initVertexArrayIdHeightMap(int width, int height) {
    glGenVertexArrays(1, &VertexArrayIdHeightMap);
    glBindVertexArray(VertexArrayIdHeightMap);
}

void fillVertexBuffersData(long long int size_g_vertex_buffer_data) {
    // An array of 3 vectors which represents 3 vertices

    // TODO: check loop
    GLfloat *p = g_vertex_buffer_data;
    for (int i = 0; i < heightTexture; i++) {
        for (int j = 0; j < widthTexture; j++) {
            // First triangle
            p[0] = j;
            p[1] = 0;
            p[2] = i;

            p[3] = j+1;
            p[4] = 0;
            p[5] = i;

            p[6] = j;
            p[7] = 0;
            p[8] = i+1;

            // Second triangle
            p[9] = j+1;
            p[10] = 0;
            p[11] = i;

            p[12] = j+1;
            p[13] = 0;
            p[14] = i+1;

            p[15] = j;
            p[16] = 0;
            p[17] = i+1;

            p += 18;
        }
    }

    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, size_g_vertex_buffer_data, &g_vertex_buffer_data[0], GL_STATIC_DRAW);

    // 1st attribute buffer : vertices
    glVertexAttribPointer(
            0,                     // attribute 0. No particular reason for 0, but must match the layout in the shader.
            3,                     // size
            GL_FLOAT,              // type
            GL_FALSE,              // normalized?
            3 * sizeof(GLfloat),   // stride
            (void *) 0             // array buffer offset
    );
    glEnableVertexAttribArray(0);
}

// Call this in while loop
void drawBuffers(int numberOfTriangles) {
    // Draw the triangles !
    glDrawArrays(GL_TRIANGLES, 0, numberOfTriangles * 18);
}

int main(int argc, char * argv[]) {

    // TODO delete this
    if (argc != 3) {
        printf("Only two texture image expected!\n");
        exit(-1);
    }

    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        exit(-1);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    window = glfwCreateWindow(1000, 1000, "CENG477 - HW3", nullptr, nullptr);

    if (nullptr == window) {
        glfwTerminate();
        exit(-1);
    }
    glfwMakeContextCurrent(window);

    glViewport(0,0,1000,1000);

    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));

        glfwTerminate();
        exit(-1);
    }

    // TODO not sure if we need ?
    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // Dark blue background TODO not sure if it is desired
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    // Depth test TODO not sure if it is desired and where to put in code
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    // Lighting
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    initHeightMap(argv[1], &widthTexture, &heightTexture);

    initVertexArrayIdHeightMap(widthTexture, heightTexture);

    initTexture(argv[2], &widthTexture, &heightTexture);

    initVertexArray(widthTexture, heightTexture);

    int numberOfTriangles = 2 * widthTexture * heightTexture;

    long long size_g_vertex_buffer_data = sizeof(GLfloat) * numberOfTriangles * 3 * 6;

    g_vertex_buffer_data = new GLfloat[numberOfTriangles * 3 * 6];

    initShaders();
    // Get a handle for our "MVP" uniform
    // Only during the initialisation

    // start !
    glUseProgram(idProgramShader);

    camera_pos = new glm::vec4((float)widthTexture / 2, (float)widthTexture / 10, (float)-widthTexture / 4, 1);
    camera_gaze = new glm::vec3(0.0f, 0.0f, 1.0f);
    camera_up = new glm::vec3(0.0f, 1.0f, 0.0f); // TODO don't know

    mvp = new glm::mat4();
    //setMVP();

    heightFactor = 10; // initially

    idMVPMatrix = static_cast<GLuint>(glGetUniformLocation(idProgramShader, "MVP"));
    glUniformMatrix4fv(idMVPMatrix, 1, GL_FALSE, glm::value_ptr(*mvp));

    idCameraPosition = static_cast<GLuint>(glGetUniformLocation(idProgramShader, "cameraPosition"));
    glUniform4fv(idCameraPosition, 1, glm::value_ptr(*camera_pos));

    idWidthTexture = static_cast<GLuint>(glGetUniformLocation(idProgramShader, "widthTexture"));
    glUniform1f(idWidthTexture, widthTexture);

    idHeightTexture = static_cast<GLuint>(glGetUniformLocation(idProgramShader, "heightTexture"));
    glUniform1f(idHeightTexture, heightTexture);

    idHeightFactor = static_cast<GLuint>(glGetUniformLocation(idProgramShader, "heightFactor"));
    glUniform1f(idHeightFactor, heightFactor);

    fillVertexBuffersData(size_g_vertex_buffer_data);

    glfwSetKeyCallback(window, keyboard); // register key callback
    glfwSetWindowSizeCallback(window, resize); // register resize callback

    while (!glfwWindowShouldClose(window)) {
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        //glUseProgram(idProgramShader);
        setMVP();

        drawBuffers(numberOfTriangles);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // disable vertex array at the end
    glDisableVertexAttribArray(0);

    // Cleanup VBO and shader
    glDeleteBuffers(1, &vertexbuffer);
    glDeleteProgram(idProgramShader);
    glDeleteVertexArrays(1, &VertexArrayId);

    // Close OpenGL window and terminate GLFW
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
