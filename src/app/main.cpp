#include <iostream>

#include "engine/core/version.h"

int main() { std::cout << "llm-inference-engine " << engine::version() << '\n'; }
