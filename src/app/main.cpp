#include <CLI/CLI.hpp>
#include <iostream>

#include "engine/core/version.h"

namespace {

void print_version() {
    std::cout << "llm-inference-engine " << engine::version() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Single-GPU LLM inference engine"};
    app.require_subcommand(0, 1);
    app.add_subcommand("version", "Print the engine version")->callback(print_version);
    CLI11_PARSE(app, argc, argv);
    if (app.get_subcommands().empty()) print_version();
}
