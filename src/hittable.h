#ifndef HITTABLE_H
#define HITTABLE_H

#include "common.h"
#include "ray.h"

class hit_record {
public:
  point3 p;
  vec3 normal;
  double t;
  bool front_face;

  void set_face_normal(const ray &r, const vec3 &outward_normal) {
    // sets the hit record normal vector
    // NOTE: outward_normal is assumed to be of unit length

    front_face = dot(r.direction(), outward_normal) < 0;
    normal = front_face ? outward_normal : -outward_normal;
  }
};

class hittable {
public:
  virtual ~hittable() = default;
  virtual bool hit(const ray &r, interval ray_t, hit_record &rec) const = 0;
};
#endif
