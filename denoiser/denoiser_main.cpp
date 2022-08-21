#include "../common/common_host.h"

#include "../common/stopwatch.h"

namespace rtc8 {

int32_t mainFunc(int32_t argc, const char* argv[]) {
	return 0;
}

}



int32_t main(int32_t argc, const char* argv[]) {
	try {
		return rtc8::mainFunc(argc, argv);
	}
	catch (const std::exception &ex) {
		hpprintf("Error: %s\n", ex.what());
		return -1;
	}
}