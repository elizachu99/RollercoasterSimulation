#include <iostream>
#include <cstring>
#include "openGLHeader.h"
#include "basicPipelineProgram.h"
using namespace std;

int BasicPipelineProgram::Init(const char * shaderBasePath, const char * vertexShaderName, const char * fragmentShaderName) 
{
  //if (BuildShadersFromFiles(shaderBasePath, "basic.vertexShader.glsl", "basic.fragmentShader.glsl") != 0)
  if (BuildShadersFromFiles(shaderBasePath, vertexShaderName, fragmentShaderName) != 0)
  {
    cout << "Failed to build the pipeline program." << endl;
    return 1;
  }
  SetShaderVariableHandles();
  cout << "Successfully built the pipeline program." << endl;
  return 0;
}

void BasicPipelineProgram::SetModelViewMatrix(const float * m)
{
  // pass "m" to the pipeline program, as the modelview matrix
  GLboolean isRowMajor = GL_FALSE;
  glUniformMatrix4fv(h_modelViewMatrix, 1, isRowMajor, m);
}

void BasicPipelineProgram::SetProjectionMatrix(const float * m) 
{
  // pass "m" to the pipeline program, as the projection matrix
  GLboolean isRowMajor = GL_FALSE;
  glUniformMatrix4fv(h_projectionMatrix, 1, isRowMajor, m);
}

int BasicPipelineProgram::SetShaderVariableHandles() 
{
  // set h_modelViewMatrix and h_projectionMatrix
  GLuint program = GetProgramHandle();
  h_modelViewMatrix = glGetUniformLocation(program,"modelViewMatrix");
  h_projectionMatrix = glGetUniformLocation(program, "projectionMatrix");

  return 0;
}

