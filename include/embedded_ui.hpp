#pragma once
#include <string>

namespace qc {
namespace ui {

/**
 * Returns the embedded index.html UI.
 * Tries disk first for instant hot-reloading if templates/index.html exists;
 * falls back to the build-time embedded version so the .bin is 100% self-contained.
 */
std::string get_index_html();

} // namespace ui
} // namespace qc
