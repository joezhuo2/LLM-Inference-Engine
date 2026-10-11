#pragma once

#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "engine/bench/metrics.h"

namespace engine {

using CsvFields = std::vector<std::pair<std::string, std::string>>;

std::string csv_escape(std::string_view field);
std::string csv_number(double value);

void write_csv_row(std::ostream& out, const CsvFields& fields);
void write_request_csv(std::ostream& out, std::span<const RequestMetrics> metrics);
CsvFields summary_fields(const Summary& summary);

}  // namespace engine
