/*
  CSCI 420 Computer Graphics, USC
  Assignment 2: Roller Coaster
  C++ starter code

  Student username: chues
*/

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <cstring>
#include "openGLHeader.h"
#include "glutHeader.h"

#include "OpenGLMatrix.h"
#include "basicPipelineProgram.h"
#include "ImageIO.h"
#include <sstream>
#include <cmath>
#include <vector>
#include "vector3.h"
#include "../external/glm/glm/gtc/type_ptr.hpp"

#ifdef WIN32
  #ifdef _DEBUG
    #pragma comment(lib, "glew32d.lib")
  #else
    #pragma comment(lib, "glew32.lib")
  #endif
#endif

#ifdef WIN32
  char shaderBasePath[1024] = SHADER_BASE_PATH;
#else
  char shaderBasePath[1024] = "../openGLHelper-starterCode";
#endif

using namespace std;

// represents one control point along the spline
struct Point 
{
  double x;
  double y;
  double z;
};

// spline struct 
// contains how many control points the spline has, and an array of control points 
struct Spline 
{
  int numControlPoints;
  Point * points;
};

int mousePos[2]; // x,y coordinate of the mouse position

int leftMouseButton = 0; // 1 if pressed, 0 if not 
int middleMouseButton = 0; // 1 if pressed, 0 if not
int rightMouseButton = 0; // 1 if pressed, 0 if not

typedef enum { ROTATE, TRANSLATE, SCALE } CONTROL_STATE;
CONTROL_STATE controlState = ROTATE;

// state of the world
float landRotate[3] = { 0.0f, 0.0f, 0.0f };
float landTranslate[3] = { 0.0f, 0.0f, 0.0f };
float landScale[3] = { 1.0f, 1.0f, 1.0f };

int windowWidth = 1280;
int windowHeight = 720;
char windowTitle[512] = "CSCI 420 homework II";

// student-defined global variables
OpenGLMatrix *matrix;
float basisMatrix[4][4];
float controlMatrix[3][4];
GLuint buffer, elementBuffer;
GLfloat delta = 2.0;
GLint axis = 2;
BasicPipelineProgram *pipelineProgram;
GLint program;
GLuint vao;
float scale = 1.0;
int channels = 0;
int countScreen = 1;
Vector3 arbitraryVector(10, 1, 1); // to compute initial normal vector
int currentU = 0;
float crossAlpha = 0.05;
bool pauseScene = false;

// objects to render, will dynamically allocate vertices
vector<float> positions, colors;
vector<float> railVertices, railColors;
vector<unsigned int> railIndices;
vector<Vector3*> tangents, normals, binormals;
int numVertices = 0;

// texture variables
GLuint groundTextureHandle, skyTextureHandle;
GLuint groundVao, skyVao;
vector<float> groundPos, skyPos;
vector<float> groundUvs, skyUvs;
int numGroundVertices, numSkyVertices;
BasicPipelineProgram *texturePipeline;
GLint textureProgram;
GLuint groundBuffer, skyBuffer;

// the spline array 
Spline * splines;
// total number of splines 
int numSplines;

// write a screenshot to the specified filename
void saveScreenshot(const char * filename)
{
  unsigned char * screenshotData = new unsigned char[windowWidth * windowHeight * 3];
  glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, screenshotData);

  ImageIO screenshotImg(windowWidth, windowHeight, 3, screenshotData);

  if (screenshotImg.save(filename, ImageIO::FORMAT_JPEG) == ImageIO::OK)
    cout << "File " << filename << " saved successfully." << endl;
  else cout << "Failed to save file " << filename << '.' << endl;

  delete [] screenshotData;
}

void setLookAt() {
  // clear transformations before LookAt
  matrix->SetMatrixMode(OpenGLMatrix::ModelView);
  matrix->LoadIdentity();
  
  // get original x, y, z
  float px = positions[currentU*3];
  float py = positions[currentU*3 + 1];
  float pz = positions[currentU*3 + 2];
  float up = 1.5f; // translates eyepos by this * up_vector so we're not right on the track
  
  matrix->LookAt(px+up*normals[currentU]->x, py+up*normals[currentU]->y, pz+up*normals[currentU]->z,
		 px+tangents[currentU]->x, py+tangents[currentU]->y, pz+tangents[currentU]->z,
		 normals[currentU]->x, normals[currentU]->y, normals[currentU]->z); 
}

void transformTextures() {
  // set camera & transform the cube a bit
  setLookAt();
  matrix->Translate(-50, 70, 50);
  matrix->Rotate(-90, 1.0, 0.0, 0.0);

  // user transformations
  matrix->Translate(landTranslate[0], landTranslate[1], landTranslate[2]);
  matrix->Rotate(landRotate[0]/10.0, 1.0, 0.0, 0.0);
  matrix->Rotate(landRotate[1]/10.0, 0.0, 1.0, 0.0);
  matrix->Rotate(landRotate[2]/10.0, 0.0, 0.0, 1.0);
  matrix->Scale(landScale[0], landScale[1], landScale[2]);
  
  // get modelview matrix
  float m[16];
  matrix->SetMatrixMode(OpenGLMatrix::ModelView);
  matrix->GetMatrix(m);

  // get projection matrix
  float p[16];
  matrix->SetMatrixMode(OpenGLMatrix::Projection);
  matrix->GetMatrix(p);

  // write matrices to texture shader
  texturePipeline->Bind();
  texturePipeline->SetModelViewMatrix(m);
  texturePipeline->SetProjectionMatrix(p);
}

void transformTrack() {
  // set the camera on the track
  setLookAt();
  
  // user transformations
  matrix->Translate(landTranslate[0], landTranslate[1], landTranslate[2]);
  matrix->Rotate(landRotate[0], 1.0, 0.0, 0.0);
  matrix->Rotate(landRotate[1], 0.0, 1.0, 0.0);
  matrix->Rotate(landRotate[2], 0.0, 0.0, 1.0);
  matrix->Scale(landScale[0], landScale[1], landScale[2]);
  
  // get modelview matrix
  float m[16];
  matrix->SetMatrixMode(OpenGLMatrix::ModelView);
  matrix->GetMatrix(m);

  // get projection matrix
  float p[16];
  matrix->SetMatrixMode(OpenGLMatrix::Projection);
  matrix->GetMatrix(p);

  // write matrices to basic shader
  pipelineProgram->Bind();
  pipelineProgram->SetModelViewMatrix(m);
  pipelineProgram->SetProjectionMatrix(p);
}

void displayFunc() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  // apply & upload transformations
  if (currentU < numVertices) {
    transformTrack();
    transformTextures();
    if (!pauseScene) currentU+=25;
  }
  
  // draw the track
  pipelineProgram->Bind();
  glBindVertexArray(vao); // bind the VAO
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBuffer);
  glDrawElements(GL_TRIANGLES, railIndices.size(), GL_UNSIGNED_INT, (void*)0); //ofset = 0
  glBindVertexArray(0); // unbind the VAO
  
  // select active texture unit
  texturePipeline->Bind();
  glActiveTexture(GL_TEXTURE0);
  // get a handle to the "textureImage" shader variable
  GLint h_textureImage = glGetUniformLocation(textureProgram, "textureImage");
  // deem the shader variable "textureImage" to read from texture unit 0
  glUniform1i(h_textureImage, 0);
  // select the texture to use
  glBindTexture(GL_TEXTURE_2D, groundTextureHandle);
  
  // draw the ground texture
  glBindVertexArray(groundVao);
  glDrawArrays(GL_TRIANGLES, 0, numGroundVertices);  
  glBindVertexArray(0);

  // draw the sky texture
  glBindTexture(GL_TEXTURE_2D, skyTextureHandle);
  glBindVertexArray(skyVao);
  glDrawArrays(GL_TRIANGLES, 0, numSkyVertices);
  glBindVertexArray(0);
  
  glutSwapBuffers();
}

void idleFunc() {
  /*
  // save the screenshots to disk (to make the animation)
  if (countScreen <= 210) {
    stringstream ss;
    ss << countScreen;
    string filename = "screenshot_" + ss.str() + ".jpg";
    saveScreenshot(filename.c_str());
    countScreen++;
  }
  */
  
  // make the screen update 
  glutPostRedisplay();
}

void reshapeFunc(int w, int h) {
  GLfloat aspect = (GLfloat) w / (GLfloat) h;
  glViewport(0, 0, w, h);

  // setup perspective matrix...
  matrix->SetMatrixMode(OpenGLMatrix::Projection);
  matrix->LoadIdentity();
  matrix->Perspective(60.0, aspect, 0.01, 1000.0); // Perspective(fovY, aspect, zNear, zFar)
  matrix->SetMatrixMode(OpenGLMatrix::ModelView);
}

void mouseMotionDragFunc(int x, int y)
{
  // mouse has moved and one of the mouse buttons is pressed (dragging)

  // the change in mouse position since the last invocation of this function
  int mousePosDelta[2] = { x - mousePos[0], y - mousePos[1] };

  switch (controlState)
  {
    // translate the landscape
    case TRANSLATE:
      if (leftMouseButton)
      {
        // control x,y translation via the left mouse button
        landTranslate[0] += mousePosDelta[0] * 0.01f;
        landTranslate[1] -= mousePosDelta[1] * 0.01f;
      }
      if (middleMouseButton)
      {
        // control z translation via the middle mouse button
        landTranslate[2] += mousePosDelta[1] * 0.01f;
      }
      break;

    // rotate the landscape
    case ROTATE:
      if (leftMouseButton)
      {
        // control x,y rotation via the left mouse button
        landRotate[0] += mousePosDelta[1];
        landRotate[1] += mousePosDelta[0];
      }
      if (middleMouseButton)
      {
        // control z rotation via the middle mouse button
        landRotate[2] += mousePosDelta[1];
      }
      break;

    // scale the landscape
    case SCALE:
      if (leftMouseButton)
      {
        // control x,y scaling via the left mouse button
        landScale[0] *= 1.0f + mousePosDelta[0] * 0.01f;
        landScale[1] *= 1.0f - mousePosDelta[1] * 0.01f;
      }
      if (middleMouseButton)
      {
        // control z scaling via the middle mouse button
        landScale[2] *= 1.0f - mousePosDelta[1] * 0.01f;
      }
      break;
  }

  // store the new mouse position
  mousePos[0] = x;
  mousePos[1] = y;
}

void mouseMotionFunc(int x, int y)
{
  // mouse has moved
  // store the new mouse position
  mousePos[0] = x;
  mousePos[1] = y;
}

void mouseButtonFunc(int button, int state, int x, int y)
{
  // a mouse button has has been pressed or depressed
  // change axis of rotation
  if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN))
    axis = 0;
  if ((button == GLUT_MIDDLE_BUTTON) && (state == GLUT_DOWN))
    axis = 1;
  if ((button == GLUT_RIGHT_BUTTON) && (state == GLUT_DOWN))
    axis = 2;

  // keep track of the mouse button state, in leftMouseButton, middleMouseButton, rightMouseButton variables
  switch (button)
  {
    case GLUT_LEFT_BUTTON:
      leftMouseButton = (state == GLUT_DOWN);
    break;

    case GLUT_MIDDLE_BUTTON:
      middleMouseButton = (state == GLUT_DOWN);
    break;

    case GLUT_RIGHT_BUTTON:
      rightMouseButton = (state == GLUT_DOWN);
    break;
  }

  // keep track of whether CTRL and SHIFT keys are pressed
  switch (glutGetModifiers())
  {
    case GLUT_ACTIVE_CTRL:
      controlState = TRANSLATE;
    break;

    case GLUT_ACTIVE_SHIFT:
      controlState = SCALE;
    break;

    // if CTRL and SHIFT are not pressed, we are in rotate mode
    default:
      controlState = ROTATE;
    break;
  }

  // store the new mouse position
  mousePos[0] = x;
  mousePos[1] = y;
}

void resetScene() {
  currentU = 0;
  landRotate[0] = 0.0f; landRotate[1] = 0.0f; landRotate[2] = 0.0f;
  landTranslate[0] = 0.0f; landTranslate[1] = 0.0f; landTranslate[2] = 0.0f;
  landScale[0] = 1.0f; landScale[1] = 1.0f; landScale[2] = 1.0f;
  pauseScene = false;
}

void deallocateVectors() {
  /* tangents, normals and binormals are vector<Vector3*> where
   * the Vector3* was dynamically allocated. Need to deallocate
   * them before this program exits or else we have memory leak */
  
  printf("\nFreeing memory...\n");
  for (int i=0; i<tangents.size(); i++) {
    delete tangents[i];
    delete normals[i];
    delete binormals[i];
  }
  printf("All dynamically allocated memory successfully deallocated.\n");
}

void keyboardFunc(unsigned char key, int x, int y)
{
  switch (key) {
    case 27: // ESC key
    case 'q':
    case 'Q':
      deallocateVectors();
      exit(0); // exit the program
      break;

    case 'r':
    case 'R':
      resetScene();
      break;

    case ' ':
      pauseScene = !pauseScene;
      break;

    case 'x':
      // take a screenshot
      saveScreenshot("screenshot.jpg");
      break;
  }
}

int loadSplines(char * argv) 
{
  char * cName = (char *) malloc(128 * sizeof(char));
  FILE * fileList;
  FILE * fileSpline;
  int iType, i = 0, j, iLength;

  // load the track file 
  fileList = fopen(argv, "r");
  if (fileList == NULL) 
  {
    printf ("can't open file\n");
    exit(1);
  }
  
  // stores the number of splines in a global variable 
  fscanf(fileList, "%d", &numSplines);

  splines = (Spline*) malloc(numSplines * sizeof(Spline));

  // reads through the spline files 
  for (j = 0; j < numSplines; j++) 
  {
    i = 0;
    fscanf(fileList, "%s", cName);
    fileSpline = fopen(cName, "r");

    if (fileSpline == NULL) 
    {
      printf ("can't open file\n");
      exit(1);
    }

    // gets length for spline file
    fscanf(fileSpline, "%d %d", &iLength, &iType);

    // allocate memory for all the points
    splines[j].points = (Point *)malloc(iLength * sizeof(Point));
    splines[j].numControlPoints = iLength;

    // saves the data to the struct
    while (fscanf(fileSpline, "%lf %lf %lf", 
	   &splines[j].points[i].x, 
	   &splines[j].points[i].y, 
	   &splines[j].points[i].z) != EOF) 
    {
      i++;
    }
  }

  free(cName);

  return 0;
}

int initTexture(const char * imageFilename, GLuint textureHandle)
{
  // read the texture image
  ImageIO img;
  ImageIO::fileFormatType imgFormat;
  ImageIO::errorType err = img.load(imageFilename, &imgFormat);

  if (err != ImageIO::OK) 
  {
    printf("Loading texture from %s failed.\n", imageFilename);
    return -1;
  }

  // check that the number of bytes is a multiple of 4
  if (img.getWidth() * img.getBytesPerPixel() % 4) 
  {
    printf("Error (%s): The width*numChannels in the loaded image must be a multiple of 4.\n", imageFilename);
    return -1;
  }

  // allocate space for an array of pixels
  int width = img.getWidth();
  int height = img.getHeight();
  unsigned char * pixelsRGBA = new unsigned char[4 * width * height]; // we will use 4 bytes per pixel, i.e., RGBA

  // fill the pixelsRGBA array with the image pixels
  memset(pixelsRGBA, 0, 4 * width * height); // set all bytes to 0
  for (int h = 0; h < height; h++)
    for (int w = 0; w < width; w++) 
    {
      // assign some default byte values (for the case where img.getBytesPerPixel() < 4)
      pixelsRGBA[4 * (h * width + w) + 0] = 0; // red
      pixelsRGBA[4 * (h * width + w) + 1] = 0; // green
      pixelsRGBA[4 * (h * width + w) + 2] = 0; // blue
      pixelsRGBA[4 * (h * width + w) + 3] = 255; // alpha channel; fully opaque

      // set the RGBA channels, based on the loaded image
      int numChannels = img.getBytesPerPixel();
      for (int c = 0; c < numChannels; c++) // only set as many channels as are available in the loaded image; the rest get the default value
        pixelsRGBA[4 * (h * width + w) + c] = img.getPixel(w, h, c);
    }

  // bind the texture
  glBindTexture(GL_TEXTURE_2D, textureHandle);

  // initialize the texture
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixelsRGBA);

  // generate the mipmaps for this texture
  glGenerateMipmap(GL_TEXTURE_2D);

  // set the texture parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // query support for anisotropic texture filtering
  GLfloat fLargest;
  glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest);
  printf("Max available anisotropic samples: %f\n", fLargest);
  // set anisotropic texture filtering
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 0.5f * fLargest);
  
  
  // query for any errors
  GLenum errCode = glGetError();
  if (errCode != 0) 
  {
    printf("Texture initialization error. Error code: %d.\n", errCode);
    return -1;
  }
  
  // de-allocate the pixel array -- it is no longer needed
  delete [] pixelsRGBA;

  return 0;
}

void initVAO() {
  // VAO (vertex array objects) to contain the VBOs
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao); // bind the VAO
  
  // bind the VBO "buffer" (must be previously created)
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  
  // get location index of the "position" shader variable
  GLint loc = glGetAttribLocation(program, "position");
  glEnableVertexAttribArray(loc);
  const void *offset = (const void*) 0;
  GLsizei stride = 0;
  GLboolean normalized = GL_FALSE;
  // set the layout of the "position" attribute data
  glVertexAttribPointer(loc, 3, GL_FLOAT, normalized, stride, offset);
  
  // get location index of the "color" shader variable
  loc = glGetAttribLocation(program, "color");
  glEnableVertexAttribArray(loc);
  offset = (const void*) (railVertices.size() * sizeof(float));
  stride = 0;
  normalized = GL_FALSE;
  // set the layout of the "color" attribute data
  glVertexAttribPointer(loc, 4, GL_FLOAT, normalized, stride, offset);

  glBindVertexArray(0); // unbind the VAO
}

void initVBO() {
  // VBO for track vertices
  glGenBuffers(1, &buffer);
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  glBufferData(GL_ARRAY_BUFFER, (railVertices.size() + railColors.size()) * sizeof(float),
	       NULL, GL_STATIC_DRAW);

  // upload position data
  glBufferSubData(GL_ARRAY_BUFFER, 0, railVertices.size() * sizeof(float), railVertices.data());

  // upload color data
  glBufferSubData(GL_ARRAY_BUFFER, railVertices.size() * sizeof(float),
		  railColors.size() * sizeof(float), railColors.data());
  
  // VBO for track indices
  glGenBuffers(1, &elementBuffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBuffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, railIndices.size() * sizeof(unsigned int),
	       &railIndices[0], GL_STATIC_DRAW);
}

void initPipelineProgram() {
  // initialize shader pipeline program
  pipelineProgram = new BasicPipelineProgram();
  pipelineProgram->Init("../openGLHelper-starterCode/", "basic.vertexShader.glsl", "basic.fragmentShader.glsl");
  pipelineProgram->Bind();
  program = pipelineProgram->GetProgramHandle();
}

void loadBasisMatrix(float s) {
  float m[16] = { -1 * s, 2.0f - s, s - 2.0f, s,
		  2 * s, s - 3.0f, 3 - 2.0f * s, -1 * s,
		  -1 * s, 0, s, 0,
		  0, 1, 0, 0 };
  //basisMatrix = glm::make_mat4(m);
  
  int i=0;
  for (int r=0; r<4; r++) {
    for (int c=0; c<4; c++) {
      basisMatrix[c][r] = m[i];
      i++;
    }
  }
}

void loadControlMatrix(Point &p1, Point &p2, Point &p3, Point &p4) {
  float m[12] = { p1.x, p1.y, p1.z,
		  p2.x, p2.y, p2.z,
		  p3.x, p3.y, p3.z,
		  p4.x, p4.y, p4.z };

  int i=0;
  for (int r=0; r<4; r++) {
    for (int c=0; c<3 ;c++) {
      controlMatrix[c][r] = m[i];
      i++;
    }
  }
}

void createVertex(const float u) {
  float us[4] = {pow(u,3), pow(u,2), u, 1.0f};
  float cur[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float tangent_us[4] = {3*pow(u,2), 2*u, 1, 0.0f};
  float tangent_cur[4] = {1.0f, 1.0f, 1.0f, 1.0f};

  // multiply [u^3, u^2, u, 1] with basis matrix to get cur[4]
  float sum = 0.0f;
  for (int c=0; c<4; c++) {
    sum = 0.0f;
    for (int r=0; r<4; r++) {
      sum += us[r] * basisMatrix[c][r];
    }
    cur[c] = sum;
  }

  // multiply cur[4] with control matrix to get [x, y, z]
  float x = 0, y = 0, z = 0;
  for (int i=0; i<4; i++) x += cur[i] * controlMatrix[0][i];
  for (int i=0; i<4; i++) y += cur[i] * controlMatrix[1][i];
  for (int i=0; i<4; i++) z += cur[i] * controlMatrix[2][i];

  // populate positions and colors
  positions.push_back(x);
  positions.push_back(y);
  positions.push_back(z);

  // do the same to tangent's u[] = derivative of given u[]
  for (int c=0; c<4; c++) {
    sum = 0.0f;
    for (int r=0; r<4; r++) {
      sum += tangent_us[r] * basisMatrix[c][r];
    }
    tangent_cur[c] = sum;
  }
  x = 0, y = 0, z = 0;
  for (int i=0; i<4; i++) x += tangent_cur[i] * controlMatrix[0][i];
  for (int i=0; i<4; i++) y += tangent_cur[i] * controlMatrix[1][i];
  for (int i=0; i<4; i++) z += tangent_cur[i] * controlMatrix[2][i];
  tangents.push_back(Vector3::unitVector(x, y, z));

  // compute normal N0 = unit(T0 x arbitraryVector), N1 = unit(B0 x T1)...
  int numTangents = tangents.size();
  Vector3 normal;
  if (numTangents <= 1) {
    normal = Vector3::crossProduct(tangents[numTangents-1], &arbitraryVector);
  } else {
    normal = Vector3::crossProduct(binormals[numTangents-2], tangents[numTangents-1]);
  }
  normals.push_back(Vector3::unitVector(normal));

  // compute binormal B0 = unit(T0 x N0), B1 = unit(T1 x N1)...
  Vector3 binormal = Vector3::crossProduct(tangents[numTangents-1], normals[numTangents-1]);
  binormals.push_back(Vector3::unitVector(binormal));
}

void pushRailIndices(GLuint i1, GLuint i2, GLuint i3,
		     GLuint i4, GLuint i5, GLuint i6) {
  railIndices.push_back(i1); railIndices.push_back(i2);
  railIndices.push_back(i3); railIndices.push_back(i4);
  railIndices.push_back(i5); railIndices.push_back(i6);
}

void allocatePoints() {
  numVertices = (splines[0].numControlPoints - 3) * 1001 ; // 1001 points for each segment

  // segments, tangents, normals, and binormals (P, T, N, B)
  for (int i=0; i<splines[0].numControlPoints-3; i++) {
    loadControlMatrix(splines[0].points[i], splines[0].points[i+1],
		      splines[0].points[i+2], splines[0].points[i+3]);
    for (float u=0.0; u < 1.001; u += 0.001) {
      createVertex(u);
    }
  }

  // railroad cross section
  for (int i=0; i<numVertices; i+=1) {
    // create 4 vertices around p(i) using P, N, and B
    glm::vec3 p0(positions[i*3], positions[i*3+1], positions[i*3+2]);
    glm::vec3 n0(normals[i]->x, normals[i]->y, normals[i]->z);
    glm::vec3 b0(binormals[i]->x, binormals[i]->y, binormals[i]->z);
    glm::vec3 v1 = p0 + crossAlpha * (b0 - n0);
    glm::vec3 v2 = p0 + crossAlpha * (n0 + b0);
    glm::vec3 v3 = p0 + crossAlpha * (n0 - b0);
    glm::vec3 v4 = p0 + crossAlpha * (-1.0f * n0 - b0);

    // add the vertices
    glm::vec3 vs[4] = {v1, v2, v3, v4};
    float shade = 0;
    for (int k=0; k<4; k++) {
      railVertices.push_back(vs[k].x);
      railVertices.push_back(vs[k].y);
      railVertices.push_back(vs[k].z);
      shade = ((float)k)/6.0 + 0.1;
      railColors.push_back(shade); railColors.push_back(shade);
      railColors.push_back(shade); railColors.push_back(1.0);
    }
  }

  // specify indices to vertices to form triangles
  GLuint vcount = 0;
  pushRailIndices(vcount, vcount+1, vcount+2, vcount, vcount+2, vcount+3); // beginning of rail
  for (int i=0; i<numVertices-1; i+=1) {
    // add the index specifications to form rectangular box
    pushRailIndices(vcount, vcount+5, vcount+4, vcount, vcount+5, vcount+1);
    pushRailIndices(vcount+1, vcount+5, vcount+2, vcount+2, vcount+5, vcount+6);
    pushRailIndices(vcount+3, vcount+2, vcount+6, vcount+6, vcount+7, vcount+3);
    pushRailIndices(vcount, vcount+3, vcount+4, vcount+3, vcount+4, vcount+7);
    vcount += 4;
  }
  pushRailIndices(vcount, vcount+1, vcount+2, vcount, vcount+2, vcount+3); // end of rail

  // init VBO and VAO
  initVBO();
  initVAO();
}

void initTexturePipeline() {
  // initialize shader pipeline program for textures
  texturePipeline = new BasicPipelineProgram();
  texturePipeline->Init("../openGLHelper-starterCode/", "texture.vertexShader.glsl", "texture.fragmentShader.glsl");
  texturePipeline->Bind();
  textureProgram = texturePipeline->GetProgramHandle();
}

void initTextureVAO(GLuint &the_vao, vector<float> &the_pos, GLuint &the_buffer) {
  // bind the VAO
  glGenVertexArrays(1, &the_vao);
  glBindVertexArray(the_vao);
  
  // bind the VBO "buffer" (must be previously created)
  glBindBuffer(GL_ARRAY_BUFFER, the_buffer);

  // get location index of the "position" shader variable
  GLint loc = glGetAttribLocation(textureProgram, "position");
  glEnableVertexAttribArray(loc);
  const void *offset = (const void*) 0;
  GLsizei stride = 0;
  GLboolean normalized = GL_FALSE;
  glVertexAttribPointer(loc, 3, GL_FLOAT, normalized, stride, offset);

  // get location index of the "texCoord" shader variable
  loc = glGetAttribLocation(textureProgram, "texCoord");
  glEnableVertexAttribArray(loc); // enable the “texCoord” attribute
  offset = (const void*) (the_pos.size() * sizeof(float));
  glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, stride, offset);

  glBindVertexArray(0); // unbind the VAO
}

void initTextureVBO(vector<float> &pos, vector<float> &uvs, GLuint &the_buffer) {
  glGenBuffers(1, &the_buffer);
  glBindBuffer(GL_ARRAY_BUFFER, the_buffer);
  // init buffer's size, but don't assign any data
  int numPosfloats = pos.size();
  int numUVfloats = uvs.size();
  glBufferData(GL_ARRAY_BUFFER, (numPosfloats + numUVfloats) * sizeof(float), NULL, GL_STATIC_DRAW);
  
  // upload position data
  glBufferSubData(GL_ARRAY_BUFFER, 0, numPosfloats * sizeof(float), pos.data());
  // upload uv data
  glBufferSubData(GL_ARRAY_BUFFER, numPosfloats * sizeof(float), numUVfloats * sizeof(float), uvs.data());
}

void addTriangle(vector<float> &pos, vector<float> &uvs,
	       float v1x, float v1y, float v1z, float v1u, float v1v,
	       float v2x, float v2y, float v2z, float v2u, float v2v,
	       float v3x, float v3y, float v3z, float v3u, float v3v) {
  // first vertex
  pos.push_back(v1x); pos.push_back(v1y); pos.push_back(v1z);
  uvs.push_back(v1u); uvs.push_back(v1v);
  // second vertex
  pos.push_back(v2x); pos.push_back(v2y); pos.push_back(v2z);
  uvs.push_back(v2u); uvs.push_back(v2v);
  // third vertex
  pos.push_back(v3x); pos.push_back(v3y); pos.push_back(v3z);
  uvs.push_back(v3u); uvs.push_back(v3v);
}

void allocateGroundTexture(int size) {
  // allocate 2 triangles for a 100 x 100 plane
  int numRepeats = 1;
  addTriangle(groundPos, groundUvs, 0, 0, 0, 0, 0,
	      size, 0, -1*size, numRepeats, numRepeats,
	      size, 0, 0, numRepeats, 0);
  addTriangle(groundPos, groundUvs, 0, 0, 0, 0, 0,
	      0, 0, -1*size, 0, numRepeats,
	      size, 0, -1*size, numRepeats, numRepeats);

  numGroundVertices = 6; // 3 vertices per triangle, 2 triangles
  
  initTextureVBO(groundPos, groundUvs, groundBuffer);
  initTextureVAO(groundVao, groundPos, groundBuffer);
}

void allocateSkyTexture(int size) {
  // allocate 10 triangles, 2 for each side of cube minus ground
  int numRepeats = 2;
  addTriangle(skyPos, skyUvs, 0, 0, 0, 0, 0,
	      size, size, 0, numRepeats, numRepeats,
	      size, 0, 0, numRepeats, 0);
  addTriangle(skyPos, skyUvs, 0, 0, 0, 0, 0,
	      0, size, 0, 0, numRepeats,
	      size, size, 0, numRepeats, numRepeats); // plane z = 0
  addTriangle(skyPos, skyUvs, 0, 0, 0, numRepeats, 0,
	      0, size, -1*size, 0, numRepeats,
	      0, 0, -1*size, 0, 0);
  addTriangle(skyPos, skyUvs, 0, 0, 0, numRepeats, 0,
	      0, size, 0, numRepeats, numRepeats,
	      0, size, -1*size, 0, numRepeats); // plane x = 0
  addTriangle(skyPos, skyUvs, 0, size, 0, 0, 0,
	      size, size, -1*size, numRepeats, numRepeats,
	      size, size, 0, numRepeats, 0);
  addTriangle(skyPos, skyUvs, 0, size, 0, 0, 0,
	      0, size, -1*size, 0, numRepeats,
	      size, size, -1*size, numRepeats, numRepeats); // plane y = 100
  addTriangle(skyPos, skyUvs, 0, 0, -1*size, 0, 0,
	      size, size, -1*size, numRepeats, numRepeats,
	      size, 0, -1*size, numRepeats, 0);
  addTriangle(skyPos, skyUvs, 0, 0, -1*size, 0, 0,
	      0, size, -1*size, 0, numRepeats,
	      size, size, -1*size, numRepeats, numRepeats); // plane z = -100
  addTriangle(skyPos, skyUvs, size, 0, 0, numRepeats, 0,
	      size, size, -1*size, 0, numRepeats,
	      size, 0, -1*size, 0, 0);
  addTriangle(skyPos, skyUvs, size, 0, 0, numRepeats, 0,
	      size, size, 0, numRepeats, numRepeats,
	      size, size, -1*size, 0, numRepeats); // plane x = 100

  numSkyVertices = 30; // 3 vertices per triangle, 10 triangles

  initTextureVBO(skyPos, skyUvs, skyBuffer);
  initTextureVAO(skyVao, skyPos, skyBuffer);
}

void initScene(int argc, char *argv[])
{
  // load the splines from the provided filename
  loadSplines(argv[1]);
  printf("Loaded %d spline(s).\n", numSplines);
  for(int i=0; i<numSplines; i++) {
    printf("Num control points in spline %d: %d.\n", i, splines[i].numControlPoints);
  }

  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  initPipelineProgram();
  initTexturePipeline();
  
  // allocate the positions & colors for the spline
  loadBasisMatrix(0.5); // s = 1/2
  allocatePoints();
  
  // load the ground and sky texture images 
  glGenTextures(1, &groundTextureHandle);
  int result = initTexture("textures/earth.jpg", groundTextureHandle);
  if (result != 0) {
    printf("Error loading the ground texture image.\n");
    exit(EXIT_FAILURE);
  }
  glGenTextures(1, &skyTextureHandle);
  result = initTexture("textures/space.jpg", skyTextureHandle);
  if (result != 0) {
    printf("Error loading the sky texture image.\n");
    exit(-1);
  }

  // initialize ground and sky textures
  allocateGroundTexture(100); // allocate a 100 x 100 plane
  allocateSkyTexture(100); // allocate a 100 x 100 cube
  
  // additional initialization
  glEnable(GL_DEPTH_TEST);
  matrix = new OpenGLMatrix();
}

int main (int argc, char ** argv)
{
  if (argc<2)
  {  
    printf ("usage: %s <trackfile>\n", argv[0]);
    exit(0);
  }

  // initialize glut
  cout << "Initializing GLUT..." << endl;
  glutInit(&argc,argv);

  cout << "Initializing OpenGL..." << endl;

  #ifdef __APPLE__
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
  #else
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
  #endif

  glutInitWindowSize(windowWidth, windowHeight);
  glutInitWindowPosition(0, 0);  
  glutCreateWindow(windowTitle);

  cout << "OpenGL Version: " << glGetString(GL_VERSION) << endl;
  cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << endl;
  cout << "Shading Language Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

  // tells glut to use a particular display function to redraw 
  glutDisplayFunc(displayFunc);
  // perform animation inside idleFunc
  glutIdleFunc(idleFunc);
  // callback for resizing the window
  glutReshapeFunc(reshapeFunc);
  // callback for mouse drags
  glutMotionFunc(mouseMotionDragFunc);
  // callback for idle mouse movement
  glutPassiveMotionFunc(mouseMotionFunc);
  // callback for mouse button changes
  glutMouseFunc(mouseButtonFunc);
  // callback for pressing the keys on the keyboard
  glutKeyboardFunc(keyboardFunc);

  // init glew
  #ifdef __APPLE__
    // nothing is needed on Apple
  #else
    // Windows, Linux
    GLint result = glewInit();
    if (result != GLEW_OK)
    {
      cout << "error: " << glewGetErrorString(result) << endl;
      exit(EXIT_FAILURE);
    }
  #endif

  // do initialization
  initScene(argc, argv);

  // keyboard directions
  cout << "\nPress <x> to take screenshot\n"
       << "<Drag> to rotate\n"
       << "Press <Ctrl and drag> to translate\n"
       << "Press <Shift and drag> to scale\n"
       << "Press <space bar> to pause\n"
       << "Press <r> to replay\n";

  // sink forever into the glut loop
  glutMainLoop();
}

