#include <climits>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <boinc/boinc_api.h>
#include <boinc/filesys.h>
#include "cpusolver.h"
#include "solverstructs.h"
#include "presolver.h"
#include "cxxopts.hpp"
#include "result_file.h"
#include "start_file.h"

#include <vector>
#include <chrono>
#include <regex>

constexpr uint8_t MAXN = 29;
constexpr uint8_t MINN = 4;
constexpr uint8_t MIN_PRE_DEPTH = 3;


static void solve_from_file(ISolver& solver, const std::string& filename) {
    file_info fi = start_file::parse_filename(filename);

    if (fi.boardsize == 0) {
        return EXIT_FAILURE;
    }

    if (fi.boardsize < MINN || fi.boardsize > MAXN) {
        std::cout << "Boardsize out of range" << std::endl;
        return EXIT_FAILURE;
    }

    if (fi.placed < MIN_PRE_DEPTH || fi.placed >= fi.boardsize) {
        std::cout << "Invalid number of placed queens" << std::endl;
        return EXIT_FAILURE;
    }

    if (fi.start_idx > fi.end_idx) {
        std::cout << "Invalid range" << std::endl;
        return EXIT_FAILURE;
    }

    uint64_t start_count = fi.end_idx - fi.start_idx + 1;

    std::vector<start_condition_t> start = start_file::load_all(filename);

    if(start_count != start.size()) {
        std::cout << "File content not matching description" << std::endl;
        return EXIT_FAILURE;
    }

    if(!solver.init(fi.boardsize, fi.placed)) {
        std::cout << "Failed to initialize solver" << std::endl;
        return EXIT_FAILURE;
    }

    uint64_t result = 0;
    for(size_t i = 0; i < start.size(); i++) {
        auto time_start = std::chrono::high_resolution_clock::now();
        result += solver.solve_subboard(start);
        auto time_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = time_end - time_start;
        std::cout << "[" << std::to_string(i) << "] Solved in " << std::to_string(elapsed) << "s" << std::endl;
        boinc_fraction_done(static_cast<double>(i)/start.size());
    }

    std::string res_filname = filename.substr(0, filename.size() - 4) + ".res";

    if(!result_file::save(result, res_filname)) {
        std::cout << "Failed to save result" << std::endl;
        return EXIT_FAILURE;
    }
}

int main(int argc, char **argv) {
    boinc_init();
    bool help = false;
    std::string presolve_file_name = "";
    try
    {
      cxxopts::Options options("m-queens2-boinc", " - a CPU solver for the N queens problem");
      options.add_options()
        ("f,file", "Use presolve file generated by 'presolver'", cxxopts::value(presolve_file_name)->default_value(""))
        ("h,help", "Print this information")
        ;

      auto result = options.parse(argc, argv);

      if (help)
      {
        options.help();
        exit(EXIT_SUCCESS);
      }

      if (result.count("file") != 1) {
          std::cout << "Only one presolve file supported" << std::endl;
          exit(EXIT_FAILURE);
      }

    } catch (const cxxopts::OptionException& e)
    {
      std::cout << "error parsing options: " << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }

    ISolver solver = cpuSolver();

    solve_from_file(solver, presolve_file_name);

  return 0;
}

