#ifndef VECTOR3_H
#define VECTOR3_H

#include <cmath>

class Vector3 {
 public:
  float x, y, z;

  Vector3() { x=0.0; y=0.0; z=0.0; }

  Vector3(float _x, float _y, float _z) {
    x = _x;
    y = _y;
    z = _z;
  }
  
  static Vector3* unitVector(const Vector3 &original) {
    return unitVector(original.x, original.y, original.z);
  }
  
  static Vector3* unitVector(const float _x, const float _y, const float _z) {
    float sumsq = _x*_x + _y*_y + _z*_z;
    float length = std::pow(sumsq, 0.5);
    Vector3 *v = new Vector3(_x/length, _y/length, _z/length);
    return v;
  }

  static Vector3 crossProduct(const Vector3 *v1, const Vector3 *v2) {
    float _x = v1->y * v2->z - v1->z * v2->y;
    float _y = v1->z * v2->x - v1->x * v2->z;
    float _z = v1->x * v2->y - v1->y * v2->x;
    return Vector3(_x, _y, _z);
  }

  Vector3 crossProduct(const Vector3 *other) {
    float _x = this->y * other->z - this->z * other->y;
    float _y = this->z * other->x - this->x * other->z;
    float _z = this->x * other->y - this->y * other->x;
    return Vector3(_x, _y, _z);
  }
};

#endif
