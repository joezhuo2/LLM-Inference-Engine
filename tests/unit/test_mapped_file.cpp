#include <gtest/gtest.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>

#include "engine/loader/mapped_file.h"
#include "support/temp_file.h"

using engine::MappedFile;
using engine::test::TempFile;

static_assert(!std::is_copy_constructible_v<MappedFile>);
static_assert(std::is_nothrow_move_constructible_v<MappedFile>);
static_assert(std::is_nothrow_move_assignable_v<MappedFile>);

TEST(MappedFile, MapsTheWholeFile) {
    std::string contents(10000, '\0');
    for (size_t i = 0; i < contents.size(); ++i) contents[i] = char(i * 31);
    const TempFile file(contents);
    const MappedFile mapped(file.path());
    ASSERT_EQ(mapped.bytes().size(), contents.size());
    EXPECT_EQ(std::memcmp(mapped.bytes().data(), contents.data(), contents.size()), 0);
}

TEST(MappedFile, EmptyFileGivesEmptySpan) {
    const TempFile file("");
    const MappedFile mapped(file.path());
    EXPECT_TRUE(mapped.bytes().empty());
}

TEST(MappedFile, MissingFileThrowsWithErrno) {
    try {
        MappedFile mapped("/nonexistent/model.safetensors");
        ADD_FAILURE() << "expected std::system_error";
    } catch (const std::system_error& e) {
        EXPECT_EQ(e.code().value(), ENOENT);
        EXPECT_NE(std::string(e.what()).find("/nonexistent/model.safetensors"), std::string::npos);
    }
}

TEST(MappedFile, DirectoryCannotBeMapped) {
    EXPECT_THROW(MappedFile{std::filesystem::temp_directory_path()}, std::system_error);
}

TEST(MappedFile, MoveTransfersTheMapping) {
    const TempFile a("first"), b("second file");
    MappedFile x(a.path()), y(b.path());
    const std::byte* p = y.bytes().data();
    x = std::move(y);
    EXPECT_EQ(x.bytes().data(), p);
    EXPECT_EQ(x.bytes().size(), 11u);
    EXPECT_TRUE(y.bytes().empty());

    MappedFile z(std::move(x));
    EXPECT_EQ(z.bytes().data(), p);
    EXPECT_TRUE(x.bytes().empty());
}
