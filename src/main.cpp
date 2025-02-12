#include "bvh.h"
#include "camera.h"
#include "common.h"
#include "hittable_list.h"
#include "quad.h"
#include "sphere.h"
#include "texture.h"
#include "vec3.h"
#include <cmath>

using std::make_shared;
using std::shared_ptr;

void bouncing_spheres() {
  hittable_list world;

  auto checker =
      make_shared<checker_texture>(0.32, color(.0, .0, .0), color(.9, .9, .9));
  world.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
                                make_shared<lambertian>(checker)));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      auto choose_mat = random_double();
      point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

      if ((center - point3(4, 0.2, 0)).length() > 0.9) {
        shared_ptr<material> sphere_material;

        if (choose_mat < 0.8) {
          // diffuse
          auto albedo = color::random() * color::random();
          sphere_material = make_shared<lambertian>(albedo);
          auto center2 = center + vec3(0, random_double(0, 0.5), 0);
          world.add(make_shared<sphere>(center, center2, 0.2, sphere_material));
        } else if (choose_mat < 0.95) {
          // metal
          auto albedo = color::random(0.5, 1);
          auto fuzz = random_double(0, 0.5);
          sphere_material = make_shared<metal>(albedo, fuzz);
          world.add(make_shared<sphere>(center, 0.2, sphere_material));
        } else {
          // glass
          sphere_material = make_shared<dielectric>(1.5);
          world.add(make_shared<sphere>(center, 0.2, sphere_material));
        }
      }
    }
  }

  auto material1 = make_shared<dielectric>(1.5);
  world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

  auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
  world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

  auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
  world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

  world = hittable_list(make_shared<bvh_node>(world));
  camera cam;

  cam.aspect_ratio = 16.0 / 9.0;
  cam.image_width = 400;
  cam.samples_per_pixel = 90;
  cam.max_depth = 10;

  cam.vfov = 20;
  cam.lookfrom = point3(13, 2, 3);
  cam.lookat = point3(0, 0, 0);
  cam.vup = vec3(0, 1, 0);

  cam.defocus_angle = 0.6;
  cam.focus_dist = 10;

  cam.render(world);
}

void checkered_spheres() {
  hittable_list world;

  auto checker =
      make_shared<checker_texture>(0.32, color(.0, .0, .0), color(.9, .9, .9));

  world.add(make_shared<sphere>(point3(0, -10, 0), 10,
                                make_shared<lambertian>(checker)));
  world.add(make_shared<sphere>(point3(0, 10, 0), 10,
                                make_shared<lambertian>(checker)));

  camera cam;

  cam.aspect_ratio = 16.0 / 9.0;
  cam.image_width = 400;
  cam.samples_per_pixel = 100;
  cam.max_depth = 50;

  cam.vfov = 20;
  cam.lookfrom = point3(13, 2, 3);
  cam.lookat = point3(0, 0, 0);
  cam.vup = vec3(0, 1, 0);

  cam.defocus_angle = 0;

  cam.render(world);
}

void perlin_spheres() {
  hittable_list world;

  auto pertext = make_shared<noise_texture>(4);
  world.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
                                make_shared<lambertian>(pertext)));
  world.add(make_shared<sphere>(point3(0, 2, 0), 2,
                                make_shared<lambertian>(pertext)));

  camera cam;

  cam.aspect_ratio = 16.0 / 9.0;
  cam.image_width = 400;
  cam.samples_per_pixel = 100;
  cam.max_depth = 50;

  cam.vfov = 20;
  cam.lookfrom = point3(13, 2, 3);
  cam.lookat = point3(0, 0, 0);
  cam.vup = vec3(0, 1, 0);

  cam.defocus_angle = 0;

  cam.render(world);
}

void earth() {
  auto earth_texture = make_shared<image_texture>("earthmap.jpg");
  auto earth_surface = make_shared<lambertian>(earth_texture);
  auto globe = make_shared<sphere>(point3(0, 0, 0), 2, earth_surface);

  camera cam;

  cam.aspect_ratio = 16.0 / 9.0;
  cam.image_width = 400;
  cam.samples_per_pixel = 100;
  cam.max_depth = 50;

  cam.vfov = 20;
  cam.lookfrom = point3(0, 0, 12);
  cam.lookat = point3(0, 0, 0);
  cam.vup = vec3(0, 1, 0);

  cam.defocus_angle = 0;

  cam.render(hittable_list(globe));
}

void cornell_box() {
  hittable_list world;

  auto red = make_shared<lambertian>(color(.65, .05, .05));
  auto white = make_shared<lambertian>(color(.73, .73, .73));
  auto green = make_shared<lambertian>(color(.12, .45, .15));
  // auto light = make_shared<diffuse_light>(color(15, 15, 15));

  world.add(make_shared<quad>(point3(555, 0, 0), vec3(0, 555, 0),
                              vec3(0, 0, 555), green));
  world.add(make_shared<quad>(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555),
                              red));
  // world.add(make_shared<quad>(point3(343, 554, 332), vec3(-130,0,0),
  // vec3(0,0,-105), light));
  world.add(make_shared<quad>(point3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555),
                              white));
  world.add(make_shared<quad>(point3(555, 555, 555), vec3(-555, 0, 0),
                              vec3(0, 0, -555), white));
  world.add(make_shared<quad>(point3(0, 0, 555), vec3(555, 0, 0),
                              vec3(0, 555, 0), white));

  camera cam;

  cam.aspect_ratio = 1.0;
  cam.image_width = 400;
  cam.samples_per_pixel = 20;
  cam.max_depth = 50;
  // cam.background        = color(0,0,0);

  cam.vfov = 40;
  cam.lookfrom = point3(278, 278, -800);
  cam.lookat = point3(278, 278, 0);
  cam.vup = vec3(0, 1, 0);

  cam.defocus_angle = 0;

  cam.render(world);
}
void quads() {
  hittable_list world;

  // Materials
  auto left_red = make_shared<lambertian>(color(1.0, 0.2, 0.2));
  auto back_green = make_shared<lambertian>(color(0.2, 1.0, 0.2));
  auto right_blue = make_shared<lambertian>(color(0.2, 0.2, 1.0));
  auto upper_orange = make_shared<lambertian>(color(1.0, 0.5, 0.0));
  auto lower_teal = make_shared<lambertian>(color(0.2, 0.8, 0.8));

  // Quads
  world.add(make_shared<quad>(point3(-3, -2, 5), vec3(0, 0, -4), vec3(0, 4, 0),
                              left_red));
  world.add(make_shared<quad>(point3(-2, -2, 0), vec3(4, 0, 0), vec3(0, 4, 0),
                              back_green));
  world.add(make_shared<quad>(point3(3, -2, 1), vec3(0, 0, 4), vec3(0, 4, 0),
                              right_blue));
  world.add(make_shared<quad>(point3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4),
                              upper_orange));
  world.add(make_shared<quad>(point3(-2, -3, 5), vec3(4, 0, 0), vec3(0, 0, -4),
                              lower_teal));

  camera cam;

  cam.aspect_ratio = 1.0;
  cam.image_width = 400;
  cam.samples_per_pixel = 100;
  cam.max_depth = 50;

  cam.vfov = 60;
  cam.lookfrom = point3(0, 0, 9);
  cam.lookat = point3(0, 0, 0);
  cam.vup = vec3(0, 1, 0);

  cam.defocus_angle = 0;

  cam.render(world);
}

int main() {
  switch (5) {
  case 1:
    bouncing_spheres();
    break;
  case 2:
    checkered_spheres();
    break;
  case 3:
    earth();
    break;
  case 4:
    perlin_spheres();
    break;
  case 5:
    quads();
    break;
  }
}
