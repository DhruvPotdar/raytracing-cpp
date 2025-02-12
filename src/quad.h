
#ifndef QUAD_H
#define QUAD_H

#include "common.h"
#include "hittable.h"
#include "hittable_list.h"
#include "interval.h"
#include "vec3.h"

class quad : public hittable {
public:
  quad(const point3 &Q, const vec3 &u, const vec3 &v, shared_ptr<material> mat)
      : Q(Q), u(u), v(v), mat(mat) {
    auto n = cross(u, v);
    normal = unit_vector(n);

    D = dot(normal, Q);
    w = n / dot(n, n);

    // std::clog << "u: " << u.x() << ", " << u.y() << ", " << u.z() <<
    // std::endl; std::clog << "v: " << v.x() << ", " << v.y() << ", " << v.z()
    // << std::endl; std::clog << "n: " << n.x() << ", " << n.y() << ", " <<
    // n.z() << std::endl;

    set_bounding_box();
  }

  virtual void set_bounding_box() {
    point3 p1 = Q;
    point3 p2 = Q + u;
    point3 p3 = Q + v;
    point3 p4 = Q + u + v;

    bbox = aabb(point3(fmin(fmin(p1.x(), p2.x()), fmin(p3.x(), p4.x())),
                       fmin(fmin(p1.y(), p2.y()), fmin(p3.y(), p4.y())),
                       fmin(fmin(p1.z(), p2.z()), fmin(p3.z(), p4.z()))),
                point3(fmax(fmax(p1.x(), p2.x()), fmax(p3.x(), p4.x())),
                       fmax(fmax(p1.y(), p2.y()), fmax(p3.y(), p4.y())),
                       fmax(fmax(p1.z(), p2.z()), fmax(p3.z(), p4.z()))));
  }

  aabb bounding_box() const override { return bbox; }

  bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
    auto denom = dot(normal, r.direction());

    if (std::fabs(denom) < 1e-6) {
      return false;
    }

    // Compute t value (distance to plane)
    auto t = (D - dot(normal, r.origin())) / denom;
    if (!ray_t.contains(t)) {
      return false;
    }

    // Compute intersection point
    auto intersection = r.at(t);
    vec3 planar_hitpt_vec = intersection - Q;

    auto alpha = dot(w, cross(planar_hitpt_vec, v));
    auto beta = dot(w, cross(u, planar_hitpt_vec));

    if (!is_interior(alpha, beta, rec)) {
      return false;
    }

    // Successful hit, fill hit record
    rec.t = t;
    rec.p = intersection;
    rec.mat = mat;
    rec.set_face_normal(r, normal);

    return true;
  }

  virtual bool is_interior(double a, double b, hit_record &rec) const {
    interval unit_interval = interval(0, 1);

    // given the hit point in plane coords,, return false if it is outside the
    // primitive otherwise set the hit record UV coordinates and return true

    if (!unit_interval.contains(a) || !unit_interval.contains(b))
      return false;

    rec.u = a;
    rec.v = b;
    return true;
  }

private:
  point3 Q;
  vec3 u, v;
  vec3 w;
  shared_ptr<material> mat;
  aabb bbox;
  vec3 normal;
  double D;
};
inline shared_ptr<hittable_list> box(const point3 &a, const point3 &b,
                                     shared_ptr<material> mat) {
  // Returns the 3D box (six sides) that contains the two opposite vertices a &
  // b.

  auto sides = make_shared<hittable_list>();

  // Construct the two opposite vertices with the minimum and maximum
  // coordinates.
  auto min = point3(std::fmin(a.x(), b.x()), std::fmin(a.y(), b.y()),
                    std::fmin(a.z(), b.z()));
  auto max = point3(std::fmax(a.x(), b.x()), std::fmax(a.y(), b.y()),
                    std::fmax(a.z(), b.z()));

  auto dx = vec3(max.x() - min.x(), 0, 0);
  auto dy = vec3(0, max.y() - min.y(), 0);
  auto dz = vec3(0, 0, max.z() - min.z());

  sides->add(make_shared<quad>(point3(min.x(), min.y(), max.z()), dx, dy,
                               mat)); // front
  sides->add(make_shared<quad>(point3(max.x(), min.y(), max.z()), -dz, dy,
                               mat)); // right
  sides->add(make_shared<quad>(point3(max.x(), min.y(), min.z()), -dx, dy,
                               mat)); // back
  sides->add(make_shared<quad>(point3(min.x(), min.y(), min.z()), dz, dy,
                               mat)); // left
  sides->add(make_shared<quad>(point3(min.x(), max.y(), max.z()), dx, -dz,
                               mat)); // top
  sides->add(make_shared<quad>(point3(min.x(), min.y(), min.z()), dx, dz,
                               mat)); // bottom

  return sides;
}
#endif
