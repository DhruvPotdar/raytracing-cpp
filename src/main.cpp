#include "camera.h"
#include "common.h"
#include "hittable_list.h"
#include "sphere.h"
#include <cmath>
#include <memory>

int main() {
  hittable_list world;
  world.add(make_shared<sphere>(point3(-0.5, -0.1, -1), 0.4));
  world.add(make_shared<sphere>(point3(0.5, -0.10, -1), 0.4));
  world.add(make_shared<sphere>(point3(0, -100.5, -1), 100));

  camera cam;
  cam.aspect_ratio = 16.0 / 9.0;
  cam.image_width = 800;
  cam.samples_per_pixel = 100;
  cam.max_depth = 100;
  cam.render(world);
}
