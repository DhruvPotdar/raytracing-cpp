#include <condition_variable>
#include <cstddef>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

using namespace std;

class ThreadPool {
public:
  // // Constructor to create a thread pool with a given number of threads
  ThreadPool(size_t num_threads = thread::hardware_concurrency()) {
    for (size_t i = 0; i < num_threads; i++) {

      threads_.emplace_back([this] {
        while (true) {
          function<void()> task;

          // Unlock the queue before executing the task so
          //  that other threads can perform enqueue tasks
          {
            // Locking the queue so that data can be sharef safely
            unique_lock<mutex> lock(queue_mutex_);

            // Waiting until there is a task to execute or the pool is stopped
            cv_.wait(lock, [this] { return !tasks_.empty() || stop_; });

            // exit the thread if pool is stopped and there are no task
            if (stop_ && tasks_.empty()) {
              return;
            }

            // Get next task
            task = std::move(tasks_.front());
            tasks_.pop();
            std::clog << "Executing task in thread: "
                      << std::this_thread::get_id() << "\n";
            task();
          }
        }
      });
    }
  }

  ~ThreadPool() {
    {
      // Lock the queue to update the stop flag safely
      std::clog << "Destroying Thread Pool\n";
      unique_lock<mutex> lock(queue_mutex_);
      stop_ = true;
    }

    // Notify all threads
    cv_.notify_all();

    // Join all worker threads to ensure they have completed their tasks
    for (auto &thread : threads_) {
      std::clog << "Joining Threads";
      thread.join();
    }
  }

  // enqueue task for executing in the thread pool
  void enqueue(function<void()> task) {
    {
      unique_lock<std::mutex> lock(queue_mutex_);
      tasks_.emplace(std::move(task));
    }
    cv_.notify_one();
  }

private:
  vector<thread> threads_;

  // Queue of Tasks
  queue<function<void()>> tasks_;

  // Mutex to sync access to shared data;
  mutex queue_mutex_;

  // Condition Variable to signal changes in the state of tasks queue
  condition_variable cv_;

  // Flag to indeicate whether the thread pool should stop or not
  bool stop_ = false;
};
