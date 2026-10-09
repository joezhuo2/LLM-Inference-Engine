#pragma once

#include <CLI/CLI.hpp>

namespace engine::app {

void add_checksum_command(CLI::App& app);
void add_generate_command(CLI::App& app);

}  // namespace engine::app
