#include <iostream>

#include "SafetensorReader.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <file.safetensors>\n";
        return 1;
    }
    std::string json = SafetensorReader::read_json_header(argv[1]);
    std::cout << "header size: " << json.size() << " bytes\n" << json << '\n';
}
