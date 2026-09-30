#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include "Buffer.hpp"

static_assert(std::is_nothrow_move_constructible_v<Buffer>,
              "Buffer must be nothrow move-constructible");
static_assert(std::is_nothrow_move_assignable_v<Buffer>, "Buffer must be nothrow move-assignable");
static_assert(!std::is_copy_constructible_v<Buffer>, "Buffer must not be copy-constructible");
static_assert(!std::is_copy_assignable_v<Buffer>, "Buffer must not be copy-assignable");
static_assert(!std::is_convertible_v<std::size_t, Buffer>,
              "Buffer(std::size_t) constructor must be explicit");

// TEST(BufferTest, AsanSanityCheck) {
//     void* ptr = std::malloc(128);
//     ASSERT_NE(ptr, nullptr) << "Allocation Failed";
//     std::memset(ptr, 0xAB, 128);
//     std::free(ptr);
//     std::free(ptr);
// }

TEST(BufferTest, ConstructionAndSize) {
    Buffer empty_buf;
    EXPECT_EQ(empty_buf.size(), 0u);
    EXPECT_EQ(empty_buf.data(), nullptr);
    EXPECT_TRUE(empty_buf.empty());

    Buffer zero_buf(0);
    EXPECT_EQ(zero_buf.size(), 0u);
    EXPECT_EQ(zero_buf.data(), nullptr);
    EXPECT_TRUE(zero_buf.empty());

    Buffer sized_buf(64);
    EXPECT_EQ(sized_buf.size(), 64u);
    ASSERT_NE(sized_buf.data(), nullptr);
    EXPECT_FALSE(sized_buf.empty());
    std::memset(sized_buf.data(), 0, 64);

    const Buffer const_buf(32);
    static_assert(std::is_same_v<decltype(std::declval<const Buffer&>().data()), const void*>,
                  "const Buffer::data() must return const void*");
    EXPECT_NE(const_buf.data(), nullptr);
    EXPECT_FALSE(const_buf.empty());
}

TEST(BufferTest, MoveConstructor) {
    Buffer src(128);
    void* original_ptr = src.data();
    ASSERT_NE(original_ptr, nullptr);

    Buffer dest(std::move(src));

    EXPECT_EQ(dest.size(), 128u);
    EXPECT_EQ(dest.data(), original_ptr);
    EXPECT_EQ(src.size(), 0u);
    EXPECT_EQ(src.data(), nullptr);
    EXPECT_TRUE(src.empty());
}

TEST(BufferTest, MoveAssignment) {
    Buffer src(256);
    Buffer dest(64);

    void* original_src_ptr = src.data();
    ASSERT_NE(original_src_ptr, nullptr);

    dest = std::move(src);

    EXPECT_EQ(dest.size(), 256u);
    EXPECT_EQ(dest.data(), original_src_ptr);
    EXPECT_EQ(src.size(), 0u);
    EXPECT_EQ(src.data(), nullptr);
    EXPECT_TRUE(src.empty());
}

TEST(BufferTest, MoveAssignEmptyCases) {
    Buffer src(128);
    Buffer empty_dest;
    void* original_ptr = src.data();
    ASSERT_NE(original_ptr, nullptr);

    empty_dest = std::move(src);
    EXPECT_EQ(empty_dest.size(), 128u);
    EXPECT_EQ(empty_dest.data(), original_ptr);
    EXPECT_FALSE(empty_dest.empty());
    EXPECT_TRUE(src.empty());

    Buffer empty_src;
    Buffer sized_dest(64);

    sized_dest = std::move(empty_src);

    EXPECT_EQ(sized_dest.size(), 0u);
    EXPECT_EQ(sized_dest.data(), nullptr);
    EXPECT_TRUE(empty_src.empty());
    EXPECT_TRUE(sized_dest.empty());
}

TEST(BufferTest, SelfMoveAssignment) {
    Buffer buf(100);
    void* original_ptr = buf.data();
    ASSERT_NE(original_ptr, nullptr);

    Buffer& alias = buf;
    buf = std::move(alias);

    EXPECT_EQ(buf.size(), 100u);
    EXPECT_EQ(buf.data(), original_ptr);
}

TEST(BufferTest, ChainedMovesAndReuse) {
    Buffer a(64);
    void* original_ptr = a.data();
    ASSERT_NE(original_ptr, nullptr);

    Buffer b = std::move(a);
    Buffer c = std::move(b);

    EXPECT_TRUE(a.empty());
    EXPECT_TRUE(b.empty());
    EXPECT_EQ(c.size(), 64u);
    EXPECT_EQ(c.data(), original_ptr);

    a = Buffer(128);
    EXPECT_EQ(a.size(), 128u);
    ASSERT_NE(a.data(), nullptr);
    EXPECT_FALSE(a.empty());
    std::memset(a.data(), 0xAA, 128);
}

TEST(BufferTest, VectorIntegration) {
    std::vector<Buffer> vec;

    for (std::size_t i = 1; i <= 100; ++i) {
        vec.emplace_back(i);
    }

    EXPECT_EQ(vec.size(), 100u);

    for (std::size_t i = 0; i < 100; i++) {
        EXPECT_EQ(vec[i].size(), i + 1);
        EXPECT_FALSE(vec[i].empty());
        ASSERT_NE(vec[i].data(), nullptr) << "at index " << i;
    }
}
