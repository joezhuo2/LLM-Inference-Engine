#include <gtest/gtest.h>

#include <filesystem>

#include "engine/loader/checksum.h"
#include "engine/loader/mapped_file.h"
#include "engine/loader/safetensors.h"
#include "support/checksum_golden.h"
#include "support/paths.h"

TEST(ChecksumGolden, FileMatchesPyTorch) {
    const auto model = engine::test::model_dir() / "model.safetensors";
    const auto golden = engine::test::golden_checksums_path();
    if (!std::filesystem::exists(model)) GTEST_SKIP() << model << " not found";
    if (!std::filesystem::exists(golden))
        GTEST_SKIP() << golden << " not found, run scripts/checksum.py";

    const engine::MappedFile file(model);
    const auto actual =
        engine::checksum_file(file.bytes(), engine::parse_safetensors_header(file.bytes()));
    engine::test::expect_matches_golden(actual, engine::test::load_golden_checksums());
}
