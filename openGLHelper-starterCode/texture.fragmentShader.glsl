#version 150

in vec2 tc;

out vec4 c;
uniform sampler2D textureImage;

void main()
{
  // compute the final fragment color by looking up the texture map
  // texture() is GLSL command to query into texture map
  c = texture(textureImage, tc); 
}

