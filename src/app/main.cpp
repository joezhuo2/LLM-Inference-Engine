#include <CLI/CLI.hpp>
#include <exception>
#include <iostream>

#include "commands.h"
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
    engine::app::add_checksum_command(app);
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    if (app.get_subcommands().empty()) print_version();
}
