#include "engine/bench/csv.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>

namespace engine {

std::string csv_escape(std::string_view field) {
    if (field.find_first_of(",\"\r\n") == std::string_view::npos) return std::string(field);
    std::string out = "\"";
    for (const char c : field) {
        if (c == '"') out += '"';
        out += c;
    }
    return out + '"';
}

std::string csv_number(double value) {
    if (std::isnan(value)) return {};
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.9g", value);
    return buf;
}

void write_csv_row(std::ostream& out, const CsvFields& fields) {
    for (size_t i = 0; i < fields.size(); ++i) out << (i ? "," : "") << csv_escape(fields[i].first);
    out << '\n';
    for (size_t i = 0; i < fields.size(); ++i)
        out << (i ? "," : "") << csv_escape(fields[i].second);
    out << '\n';
}

void write_request_csv(std::ostream& out, std::span<const RequestMetrics> metrics) {
    out << "id,prompt_tokens,output_tokens,preemptions,arrival_s,ttft_s,tpot_s,latency_s\n";
    for (const auto& m : metrics)
        out << m.id << ',' << m.prompt_tokens << ',' << m.output_tokens << ',' << m.preemptions
            << ',' << csv_number(m.arrival_s) << ',' << csv_number(m.ttft_s) << ','
            << csv_number(m.tpot_s) << ',' << csv_number(m.latency_s) << '\n';
}

CsvFields summary_fields(const Summary& s) {
    CsvFields f = {
        {"requests", std::to_string(s.requests)},
        {"prompt_tokens", std::to_string(s.prompt_tokens)},
        {"output_tokens", std::to_string(s.output_tokens)},
        {"wall_s", csv_number(s.wall_s)},
        {"request_throughput", csv_number(s.request_throughput)},
        {"output_throughput", csv_number(s.output_throughput)},
    };
    for (const auto& [name, stat] : {std::pair{"ttft_s", s.ttft_s}, std::pair{"tpot_s", s.tpot_s},
                                     std::pair{"latency_s", s.latency_s}}) {
        const std::string base(name);
        f.emplace_back(base + "_mean", csv_number(stat.mean));
        f.emplace_back(base + "_p50", csv_number(stat.p50));
        f.emplace_back(base + "_p90", csv_number(stat.p90));
        f.emplace_back(base + "_p99", csv_number(stat.p99));
    }
    return f;
}

}  // namespace engine
