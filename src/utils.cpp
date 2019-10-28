#include <algorithm>
#include <iostream>
#include "utils.h"

using namespace std;

const chrono::milliseconds Millisecond(1);
const chrono::milliseconds ZeroMilliseconds(0);

Rate::Rate(const int rate) {
  ms_per_frame = 1000 / rate;
  last_timestep = chrono::system_clock::now();
}

void Rate::sleep() {
  chrono::system_clock::time_point now = chrono::system_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_timestep);
  auto sleep_for = Millisecond * ms_per_frame - diff;
  if (ZeroMilliseconds > sleep_for) {
    sleep_for = ZeroMilliseconds;
  }
  std::this_thread::sleep_for(sleep_for);
  last_timestep = chrono::system_clock::now();
}

