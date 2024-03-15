# =======================================
# demo
# =======================================
# wosolver
add_subdirectory(dependencies/polyscope)
add_executable(test_wosolver wosolver.cu)
target_link_libraries(test_wosolver cuwos polyscope)

# test
# get google test
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_bvh test/bvh.cu)
target_link_libraries(test_bvh GTest::gtest_main GTest::gmock_main cuwos)

include(GoogleTest)
gtest_discover_tests(test_bvh)
