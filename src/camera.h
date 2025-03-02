#ifndef CAMERA_H
#define CAMERA_H

#include "common.h"

#include "color.h"
#include "hittable.h"
#include "interval.h"
#include "material.h"
#include "ray.h"
#include "thread_pool.h"
#include "vec3.h"

#include <cassert>
#include <future>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <thread>
#include <vector>

static std::mutex mtx;
class camera {
public:
  double aspect_ratio = 1.0;  // Ratio of image width over height
  int image_width = 100;      // Rendered image width in pixel count
  int samples_per_pixel = 10; // Count of random samples for each pixel

  // Calculate total number of pixels (width * height)
  int cam_vec_length;

  struct PixelData {
    color pixel_color;
    std::vector<ray> rays;
  };
  std::vector<PixelData> camera_vec;

  std::atomic<int> completed_chunks{0};
  std::mutex completion_mutex;
  std::condition_variable completion_cv;

  int max_depth = 10; // Max number of ray bounces in the scene

  int vfov = 90;                     // vertical
  point3 lookfrom = point3(0, 0, 0); // Point cam is looking from
  point3 lookat = point3(0, 0, -1);  // Point camera is looking to
  vec3 vup = vec3(0, 1, 0);          // Cam relative up direction
  //
  double defocus_angle = 0; // Variation of angle of rays through each pixel
  double focus_dist =
      10; // distance fram camera lookfrom point to plane of perfect focus
  color background; // Scene bg color

  std::vector<std::future<void>> future_vec;

  void initialize() {
    image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    pixel_sample_scale = 1.0 / samples_per_pixel;
    cam_vec_length = image_height * image_width;

    center = lookfrom;

    // Determine viewport dimensions.
    auto theta = degrees_to_radians(vfov);
    auto h = std::tan(theta / 2);
    auto viewport_height = 2 * h * focus_dist;
    auto viewport_width =
        viewport_height * (double(image_width) / image_height);

    // Calculate u,v,w unit basis vectors for the camera coordinate frame
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    // Calculate the vectors across the horizontal and down the vertical
    // viewport edges.
    vec3 viewport_u = viewport_width * u;
    vec3 viewport_v = viewport_height * -v;

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    pixel_delta_u = viewport_u / image_width;
    pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left =
        center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Calculate the camera defocus disk basis vectors
    auto defocus_radius =
        focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
    defocus_disk_u = u * defocus_radius;
    defocus_disk_v = v * defocus_radius;
  }

  void generate_samples(color &pixel_color, const int index,
                        const hittable &world) {
    int i, j;
    index_to_2d(index, image_width, i, j);

    color accumulator; // Local accumulator to avoid frequent locking
    for (int sample = 0; sample < samples_per_pixel; sample++) {
      auto r = get_ray(i, j);

      auto ray_col = ray_color(r, max_depth, world);
      accumulator += ray_col;
    }

    // Update shared resource with a single lock
    {
      // std::lock_guard<std::mutex> lock(mtx);
      // std::unique_lock<std::mutex> lock(mtx);
      pixel_color += accumulator;
    }
  }

  void parallel_render(const hittable &world) {
    initialize();

    std::vector<PixelData> camera_vec(
        cam_vec_length,
        PixelData{color(), std::vector<ray>(samples_per_pixel)});

    int num_workers = std::thread::hardware_concurrency();
    ThreadPool pool(num_workers);
    int chunk_size =
        (cam_vec_length + num_workers - 1) / num_workers; // Round up

    std::atomic<int> completed_chunks = 0;

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int worker = 0; worker < num_workers; worker++) {

      int start_index = worker * chunk_size;
      int end_index = std::min(start_index + chunk_size, cam_vec_length);

      assert(start_index >= 0 && end_index <= cam_vec_length);

      // std::clog << "Worker " << worker << ": Start " << start_index << ",End
      // "
      //           << end_index << "\n";
      pool.enqueue([this, start_index, end_index, &world, &completed_chunks,
                    num_workers, &camera_vec]() {
        try {

          // Process each pixel in this chunk
          for (int index = start_index; index < end_index; ++index) {
            generate_samples(camera_vec[index].pixel_color, index, world);
          }

          std::clog << "Chunk completed: " << completed_chunks.load() << "/"
                    << num_workers << "\n\n";

        } catch (const std::exception &e) {
          std::cerr << "Error in thread: " << e.what() << std::endl;
        }

        if (++completed_chunks == num_workers) {
          std::lock_guard<std::mutex> lk(completion_mutex);
          completion_cv.notify_one();
        }
      });
    }
    // Synchronization barrier
    {
      std::unique_lock<std::mutex> lk(completion_mutex);
      completion_cv.wait(lk, [&] { return completed_chunks == num_workers; });
    }

    // Write colors serially because doing that in parallel will cause problems
    for (const auto &pixel : camera_vec) {
      write_color(std::cout, pixel_sample_scale * pixel.pixel_color);
    }

    std::clog << "\rDone.                 \n";
    return;
  }

  void render(const hittable &world) {
    initialize();

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    // #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int j = 0; j < image_height; j++) {
      std::clog << "\rScanlines remaining: " << (image_height - j) << ' '
                << std::flush;
      for (int i = 0; i < image_width; i++) {
        color pixel_color(0, 0, 0);

        for (int sample = 0; sample < samples_per_pixel; sample++) {
          ray r = get_ray(i, j);
          pixel_color += ray_color(r, max_depth, world);
        }
        write_color(std::cout, pixel_sample_scale * pixel_color);
      }
    }

    // std::clog << "\rDone.                 \n";
  }

  static void index_to_2d(int index, int width, int &i, int &j) {
    i = index % width; // Column (x)
    j = index / width; // Row (y)
  }

  static int two_d_to_index(int i, int j, int width) { return j * width + i; }

  // void zender(const hittable &world) {
  //   // Use std::async
  //   initialize();
  //
  //   auto cam_vec_length = image_width * image_height;
  //
  //   std::vector<PixelData> camera_vec(
  //       cam_vec_length,
  //       PixelData{color(), std::vector<ray>(samples_per_pixel)});
  //
  //   std::clog << "accumulator " << camera_vec[0].pixel_color.x() << " "
  //             << camera_vec[0].pixel_color.y() << " "
  //             << camera_vec[0].pixel_color.z() << "\n";
  //   std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
  //
  //   for (int index = 0; index < cam_vec_length; index++) {
  //
  //     // if (cam_vec_length - index == 1)
  //     //   break;
  //     std::clog << "\r Rays remaining: " << cam_vec_length - index << "/"
  //               << cam_vec_length << '\n'
  //               << std::flush;
  //
  //     future_vec.push_back(
  //         std::async(std::launch::async, [this, &camera_vec, index, &world]()
  //         {
  //           this->generate_samples(camera_vec[index].pixel_color, index,
  //           world);
  //         }));
  //     // generate_samples(camera_vec[index], index, world);
  //   }
  //
  //   //  Wait for the futures to complete
  //   std::clog << "Checking for futures \n";
  //   for (auto &future : future_vec) {
  //     future.wait();
  //   }
  //   // Write the colors serially because doing that in parallel will cause
  //   // problems in image rendering
  //   for (const auto &pixel : camera_vec) {
  //     write_color(std::cout, pixel_sample_scale * pixel.pixel_color);
  //   }
  //
  //   std::clog << "\rDone.                 \n";
  // }
  //
  // void bender(const hittable &world) {
  //   initialize();
  //
  //   // Create a pixel data vector which allows me to do 2 things
  //   // 1. Iterate over the vector of images since it is flat
  //   // 2. Perform multiple ray hits as once since the samples for each ray
  //   are
  //   // also vectors so they can be parallelized as well
  //   std::vector<PixelData> camera_vec(
  //       image_width * image_height,
  //       PixelData{color(), std::vector<ray>(samples_per_pixel)});
  //
  //   std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
  //   std::for_each(std::execution::parallel_unsequenced_policy(),
  //                 camera_vec.begin(), camera_vec.end(), [&](PixelData &pixel)
  //                 {
  //                   // Very smart way to calculate the index it seems
  //                   size_t index = &pixel - &camera_vec[0];
  //                   int i = index % image_width;
  //                   int j = index / image_width;
  //
  //                   // std::clog << "\rScanlines remaining: " <<
  //                   (image_height -
  //                   // j)
  //                   //           << ' ' << std::flush;
  //                   // Process each ray in the pixel
  //
  //                   std::for_each(
  //                       // std::execution::parallel_unsequenced_policy(),
  //                       pixel.rays.begin(), pixel.rays.end(), [&](ray &r) {
  //                         // Initialize or modify each ray
  //                         r = get_ray(i, j);
  //                         // std::lock_guard<std::mutex> lock(mtx);
  //                         pixel.pixel_color += ray_color(r, max_depth,
  //                         world);
  //                       });
  //                 });
  //
  //   // Write the colors serially because doing that in parallel will cause
  //   // problems in image rendering
  //   for (const auto &pixel : camera_vec) {
  //     write_color(std::cout, pixel_sample_scale * pixel.pixel_color);
  //   }
  // }

private:
  int image_height; // Rendered image height
  point3 center;    // Camera center
  double pixel_sample_scale;
  point3 pixel00_loc;  // Location of pixel 0, 0
  vec3 pixel_delta_u;  // Offset to pixel to the right
  vec3 pixel_delta_v;  // Offset to pixel below
  vec3 u, v, w;        // Camera frame basis vectors
  vec3 defocus_disk_v; // Defocus disk vertical radius
  vec3 defocus_disk_u; // Defocus disk horizontal radius

  ray get_ray(int i, int j) {
    // Construct a camera ray from the defocus disk and directed at a randomly
    // sampled point in the pixel location i, j

    auto offset = sample_square();
    auto pixel_sample = pixel00_loc + ((j + offset.x()) * pixel_delta_v) +
                        ((i + offset.y()) * pixel_delta_u);
    auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
    auto ray_direction = pixel_sample - ray_origin;
    auto ray_time = random_double();

    return ray(ray_origin, ray_direction, ray_time);
  }

  color ray_color(const ray &r, int depth, const hittable &world) const {
    if (depth <= 0) {
      return color(0, 0, 0);
    }
    hit_record rec;

    if (!world.hit(r, interval(0.001, infinity), rec))
      return background;

    ray scattered;
    color attenuation;
    color color_from_emission = rec.mat->emitted(rec.u, rec.v, rec.p);

    if (!rec.mat->scatter(r, rec, attenuation, scattered))
      return color_from_emission;

    color color_from_scatter =
        attenuation * ray_color(scattered, depth - 1, world);

    return color_from_emission + color_from_scatter;
  }

  vec3 sample_square() const {
    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit
    // square.
    return vec3(random_double() - 0.5, random_double() - 0.5, 0);
  }
  point3 defocus_disk_sample() const {
    // Return a random point onthe camera defocus dist
    auto p = random_in_unit_disk();
    return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
  }
};
#endif
