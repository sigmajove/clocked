// A program to estimate the probability that, in a randomly generated
// tetrahedron, the longest and shortest sides do not share a vertex.

// Monte Carlo techinques require an enormous amount of computing power.
// This program runs for 6 hours, computing as many samples as it can
// in that time. The number of samples will depend upon the horsepower
// of the machine.
// 
// On a 3.4 GHz x86 with 8 hypercores, it computed 5.5 trillion samples,
// producing the result 29.2495% +/- .0003%.

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

// We use as many worker threads as will help.
const unsigned int num_threads = std::thread::hardware_concurrency();

// We use a batch_size that is a power of two in the (unconfirmed) hope
// that it will speed up the division at the end of Batch.
constexpr size_t batch_size = 1 << 14;
static_assert(batch_size == 16'384);

// Generates batch_size random tetrahedra and counts how many
// of them have the longest side opposite the shortest side.
// dist returns a random number in the range [0.0, 1.1].
// Returns the result as a percentage in the range [0.0, 100.0].
double Batch(const std::function<double()>& dist) {
  size_t count = 0;

  for (uint64_t i = 0; i < batch_size; ++i) {
    // We can assume without loss of generality that the longest edge is a,
    // and that a = 1.
    constexpr double a = 1.0;

    double b, c, a1, b1, c1;  // The other five edges.
    for (;;) {
      // Keep looping until we find a valid tetrahedron.
      b = dist();
      c = dist();

      // Make sure (a=1, b, c) satisfy the triangle inequality.
      // If they don't, we can adjust b and c so they do without
      // changing the result. If b is a uniform distribution, then
      // 1 - b is also a uniform distribution.
      if (b + c < a) {
        const double bb = 1.0 - b;
        const double cc = 1.0 - c;
        b = bb;
        c = cc;
      }

      b1 = dist();
      c1 = dist();

      // We can use the same trick with (a=1, b1, c1).
      if (b1 + c1 < a) {
        const double bb = 1.0 - b1;
        const double cc = 1.0 - c1;
        b1 = bb;
        c1 = cc;
      }

      a1 = dist();

      // Make sure (a1, b1, c) satisfy the triangle inequality.
      // If they don't, we are out of wiggle room, and have to start over.
      // We might be tempted to set c to a random number between
      // abs(a1 - b1) and a1 + b1, but then c would not be an independent
      // variable.
      if (!(std::abs(a1 - b1) <= c && c <= a1 + b1)) continue;

      // Test whether the six sides form a valid tetrahedron.
      // The test is nonobvious. For an explantion, see
      // https://www.dropbox.com/scl/fi/h1gmhfkvdpk31f3lqlv9l/Programming-is-a-Challenge.pdf?rlkey=iiao4gq2pwq1l71b22ilexhw7&dl=0
      b = b * b;
      c = c * c;
      a1 = a1 * a1;
      b1 = b1 * b1;
      c1 = c1 * c1;
      const double a2 = b1 + c1 - a;
      const double b2 = a1 + c1 - b;
      const double c2 = a1 + b1 - c;
      if (4 * a1 * b1 * c1 + a2 * b2 * c2 >=
          a1 * a2 * a2 + b1 * b2 * b2 + c1 * c2 * c2)
        break;  // success
    }

    // We have a valid tetrahederon. Count it if a1 is the shortest side.
    if (a1 <= b && a1 <= c && a1 <= b1 && a1 <= c1) {
      ++count;
    }
  }

  // Return the percentage of random tetrahedra with shortest and longest
  // sides opposing.
  return (100.0 * count) / batch_size;
}

// GlobalData is shared by all the worker threads.
struct GlobalData {
  TimePoint stop_time;  // logically a constant.

  // Notified when data is added to queue.
  std::condition_variable data_sent;

  // Notified when the queue is empty.
  std::condition_variable queue_drained;

  std::mutex lock;           // protects the following members
  std::deque<double> queue;  // where the answers go
  unsigned int num_threads_running = num_threads;
};

// A class for keeping an average of the data points
// read by the consumer thread. It distributes the data
// points into multiple buckets so we can estimate an error bar.
class RunningAverage {
 public:
  static constexpr size_t num_buckets = 10;

  // Adds num_buckets data points to the averages./
  // Requires results.size() == num_buckets.
  void Add(const std::vector<double>& results) {
    for (size_t i = 0; i < sum_.size(); ++i) {
      sum_[i] += results[i];
    }
    ++count_;
  }

  // Prints on stdout the data accumulated so far.
  void Aggregate();

 private:
  std::array<double, num_buckets> sum_;  // The sums of the data points.
  uint64_t count_ = 0;  // The number of times Add has been called.
};


void RunningAverage::Aggregate() {
  double smallest = std::numeric_limits<double>::max();
  double largest = -smallest;
  double total = 0.0;

  for (double s : sum_) {
    const double r = s / count_;
    smallest = std::min(r, smallest);
    largest = std::max(r, largest);
    total += r;
  }
  const double average = total / sum_.size();
  const double tolerance = std::max(largest - average, average - smallest);
  printf("%.5f%% +/- %.5f%% after %.1f billion samples\n", average, tolerance,
         static_cast<double>(count_ * batch_size * num_buckets * num_threads) /
             1'000'000'000.0);
  fflush(stdout);  // In case the program dies before it completes.
}

// The code run by a worker thread.
// It generates data points and writes them to global->queue.
void WorkerThread(GlobalData* global) {
  std::random_device device;

  // A 64-bit Mersenne Twister with a random seed.
  std::mt19937_64 generator(device());
  std::uniform_real_distribution<double> unit_dist(0.0, 1.0);

  // Package the random generator into a functional that can be passed to Batch.
  const auto dist = [&generator, &unit_dist] { return unit_dist(generator); };

  // Keep generating data points util we reach the stop time.
  while (std::chrono::steady_clock::now() < global->stop_time) {
    const double value = Batch(dist);  // Compute a data point.
    std::unique_lock<std::mutex> hold(global->lock);
    // There are several producers and only one consumer.
    // Keep the queue from using too much memory if the producers
    // run faster than the consumer can handle.
    while (global->queue.size() >= 100) {
      // Wait for the consumer to catch up.
      global->queue_drained.wait(hold);
    }

    global->queue.push_front(value);
    const bool should_notify =
        global->queue.size() >= RunningAverage::num_buckets;
    hold.unlock();
    if (should_notify) {
      // Alert the consumer in case it is waiting for data.
      global->data_sent.notify_one();
    }
  }

  // We have generated all the data we are going to.
  // No longer count ourselves as a running thread.
  global->lock.lock();
  const bool should_notify = --global->num_threads_running == 0;
  global->lock.unlock();
  if (should_notify) {
    // Notify the consumer in case it is waiting for all the
    // threads to terminate.
    global->data_sent.notify_one();
  }
}

void RunWorkerThreads() {
  // A 64-bit Mersenne Twister with a random seed.
  std::random_device device;
  std::mt19937_64 generator(device());

  RunningAverage average;  // The answer goes here.

  const TimePoint start_time = std::chrono::steady_clock::now();

  // How often to display partial results.
  const std::chrono::minutes tick = std::chrono::minutes(15);

  // When to next display partial results.
  TimePoint next_tick = start_time + tick;

  // next_tick = start_time + num_ticks * tick.
  size_t num_ticks = 1;

  GlobalData global;

  // Run for six hours.
  global.stop_time = start_time + std::chrono::hours(6);

  // Kick off all the worker threads.
  std::vector<std::thread> threads;
  for (unsigned int i = 0; i < num_threads; ++i) {
    threads.emplace_back(WorkerThread, &global);
  }

  std::unique_lock<std::mutex> hold(global.lock);
  for (;;) {
    if (global.queue.size() >= RunningAverage::num_buckets) {
      // Pull num_buckets data points off the queue and
      // add them to the running average.
      std::vector<double> results;
      results.reserve(RunningAverage::num_buckets);
      for (size_t i = 0; i < RunningAverage::num_buckets; ++i) {
        results.push_back(global.queue.back());
        global.queue.pop_back();
      }

      // The intent is to avoid patterns in the data.
      // Probably a silly idea.
      std::shuffle(results.begin(), results.end(), generator);

      average.Add(results);
     
      if (std::chrono::steady_clock::now() >= next_tick) {
         // Periodically display how things are going.
        average.Aggregate();
        next_tick = start_time + (++num_ticks) * tick;
      }
    } else if (global.num_threads_running == 0) {
      // If there is no data left (we ignore fewer then num_buckets
      // at the end), it's time to stop.
      break;
    } else {
      // The queue has been completely drained. Notify the producers
      // to start making more data for us to read.
      global.queue_drained.notify_all();

      // Wait until there is at least num_buckets data points
      // to process.
      global.data_sent.wait(hold);
    }
  }

  // Wait for all the worker threads to terminate.
  for (std::thread& t : threads) t.join();

  // Display the final result.
  average.Aggregate();

  printf("Finished\n");  // That's all, folks!
}

int main() { RunWorkerThreads(); }