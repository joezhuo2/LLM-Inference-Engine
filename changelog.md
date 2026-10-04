# Changelog

Changes are grouped by branch, newest first, in the order the branches merge into `main`. Each entry is tagged with its part of the delegation: Part A and Part B entries are written by Claude and reviewed by Joe, and Part C entries are written by Joe.

## build/m1-scaffolding

- Add CLI11 (found on the system first, fetched otherwise) and turn `engine` into a subcommand CLI with a `version` subcommand; running it with no arguments still prints the version. (Part A)
- Rename the `json` FetchContent dependency to `nlohmann_json` so CMake stops warning about a package name mismatch on every configure. (Part A)
- CI runs `ctest -LE todo_joe` so tests for unimplemented Part C components do not fail the build, installs CLI11 from apt and Homebrew, and adds a job that byte-compiles every tracked Python file. (Part A)
- `.clangd` reads `build/debug/compile_commands.json` directly, so the root symlink is no longer needed. (Part A)
- Add `scripts/requirements.txt` pinned to the `~/.venvs/llmie` versions, with the cu128 index for an sm_120 torch build. (Part A)
