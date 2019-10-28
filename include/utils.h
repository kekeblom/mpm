#include <chrono>
#include <thread>

class Rate {
  private:
  unsigned int ms_per_frame;
  std::chrono::system_clock::time_point last_timestep;

  public:
  Rate(const int rate);
  void sleep();
};

