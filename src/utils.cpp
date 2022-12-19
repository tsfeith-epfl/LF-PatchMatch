//
// Created by tsfeith on 13/12/22.
//

#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>
#include "cnpy.h"

using namespace std;

vector<string> get_scene_names(const string &scene_dir) {
    vector<string> scene_names;
    for (const auto &entry: filesystem::directory_iterator(scene_dir)) {
        if (entry.path().extension() != ".png") {
            continue;
        }
        scene_names.push_back(entry.path().filename());
    }
    sort(scene_names.begin(), scene_names.end());
    return scene_names;
}

vector<vector<vector<vector<vector<uint8_t>>>>> get_scene_grid(const string &directory_path) {
    vector<vector<vector<vector<vector<uint8_t>>>>> scene_grid;
    vector<filesystem::directory_entry> entries;
    for (const auto &entry: filesystem::directory_iterator(directory_path)) {
        if (entry.path().extension() != ".png") {
            continue;
        }
        entries.push_back(entry);
    }
    sort(entries.begin(), entries.end(),
         [](const filesystem::directory_entry &a, const filesystem::directory_entry &b) {
             return a.path().filename() < b.path().filename();
         });
    for (const auto &entry: entries) {
        string filename = entry.path().filename().string();
        int row = stoi(filename.substr(filename.length() - 9, 2));
        if (row == scene_grid.size()) {
            scene_grid.emplace_back();
        }
        cv::Mat image = cv::imread(entry.path().string());

        vector<vector<vector<uint8_t>>> data = vector<vector<vector<uint8_t>>>(3, vector<vector<uint8_t>>(image.rows,
                                                                                                          vector<uint8_t>(
                                                                                                                  image.cols)));
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
                data[0][i][j] = pixel[2];
                data[1][i][j] = pixel[1];
                data[2][i][j] = pixel[0];
            }
        }
        scene_grid[row].push_back(data);
    }
    return scene_grid;
}

vector<vector<vector<uint8_t>>> limited_insert(vector<vector<vector<uint8_t>>> matching_patches,
                                               vector<int> &differences,
                                               vector<vector<vector<uint8_t>>> best_patch,
                                               int min_diff,
                                               int max_size) {
    if (differences.size() < max_size) {
        matching_patches.emplace_back(best_patch[0]);
        matching_patches.emplace_back(best_patch[1]);
        matching_patches.emplace_back(best_patch[2]);
        differences.emplace_back(min_diff);
        if (differences.size() == max_size) {
            for (int i = 0; i < differences.size(); i++) {
                for (int j = i + 1; j < differences.size(); j++) {
                    if (differences[i] > differences[j]) {
                        int temp = differences[i];
                        differences[i] = differences[j];
                        differences[j] = temp;

                        vector<vector<uint8_t>> temp_patch1 = matching_patches[3 * i];
                        vector<vector<uint8_t>> temp_patch2 = matching_patches[3 * i + 1];
                        vector<vector<uint8_t>> temp_patch3 = matching_patches[3 * i + 2];
                        matching_patches[3 * i] = matching_patches[3 * j];
                        matching_patches[3 * i + 1] = matching_patches[3 * j + 1];
                        matching_patches[3 * i + 2] = matching_patches[3 * j + 2];
                        matching_patches[3 * j] = temp_patch1;
                        matching_patches[3 * j + 1] = temp_patch2;
                        matching_patches[3 * j + 2] = temp_patch3;
                    }
                }
            }
        }
    } else {
        if (min_diff >= differences[max_size - 1]) {
            return matching_patches;
        }
        int left = 0;
        int right = max_size - 1;
        while (left < right) {
            int mid = (left + right) / 2;
            if (differences[mid] < min_diff) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        differences.insert(differences.begin() + left, min_diff);
        differences.pop_back();

        matching_patches.insert(matching_patches.begin() + 3 * left, best_patch[0]);
        matching_patches.insert(matching_patches.begin() + 3 * left + 1, best_patch[1]);
        matching_patches.insert(matching_patches.begin() + 3 * left + 2, best_patch[2]);
        matching_patches.pop_back();
        matching_patches.pop_back();
        matching_patches.pop_back();
    }
    return matching_patches;
}

vector<vector<vector<uint8_t>>> get_matching_patches(vector<vector<vector<vector<vector<uint8_t>>>>> grid,
                                                     int i,
                                                     int j,
                                                     int start_row,
                                                     int start_col,
                                                     int patch_size,
                                                     int num_similar,
                                                     int search_stride,
                                                     int roi) {
    vector<int> patchsize = vector<int>(2, 0);
    patchsize[0] = min((int) grid[i][j][0].size() - start_row, patch_size);
    patchsize[1] = min((int) grid[i][j][0][0].size() - start_col, patch_size);

    vector<int> grid_h, grid_v;
    for (int k = 0; k < grid.size(); k++) {
        if (k == i) {
            continue;
        }
        grid_v.emplace_back(k);
    }
    for (int k = 0; k < grid[i].size(); k++) {
        if (k == j) {
            continue;
        }
        grid_h.emplace_back(k);
    }


    // create empty vectors
    vector<vector<vector<uint8_t>>> matching_patches;
    vector<int> differences;
    vector<int> prev_position = {start_row, start_col};

    vector<int> search_space;
    for (auto h: grid_h) {
        int min_difference = 1e8;
        // new_image = grid[i][h]
        search_space.clear();
        for (int k = -roi; k <= roi; k++) {
            if (prev_position[1] + k * search_stride >= 0 and
                prev_position[1] + k * search_stride + patchsize[1] <= grid[i][h][0][0].size()) {
                int range_h = prev_position[1] + k * search_stride;
                search_space.emplace_back(range_h);
            }
        }

        for (auto pos: search_space) {
            int difference = 0;
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < patchsize[0]; l++) {
                    for (int m = 0; m < patchsize[1]; m++) {
                        difference += abs(grid[i][h][k][prev_position[0] + l][pos + m] - grid[i][j][k][start_row + l][start_col + m]);
                    }
                }
            }
            if (difference < min_difference) {
                min_difference = difference;
                prev_position = {prev_position[0], pos};
            }
        }
        vector<vector<vector<uint8_t>>> best_patch = vector<vector<vector<uint8_t>>>(3, vector<vector<uint8_t>>(
                patchsize[0], vector<uint8_t>(patchsize[1])));
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < patchsize[0]; l++) {
                for (int m = 0; m < patchsize[1]; m++) {
                    best_patch[k][l][m] = grid[i][h][k][prev_position[0] + l][prev_position[1] + m];
                }
            }
        }
        matching_patches = limited_insert(matching_patches, differences, best_patch, min_difference, num_similar);
    }

    for (auto v: grid_v) {
        int min_difference = 1e8;
        // new_image = grid[v][h]
        search_space.clear();
        for (int k = -roi; k <= roi; k++) {
            if (prev_position[0] + k * search_stride >= 0 and
                prev_position[0] + k * search_stride + patchsize[0] <= grid[v][j][0].size()) {
                int range_v = prev_position[0] + k * search_stride;
                search_space.emplace_back(range_v);
            }
        }
        for (auto pos: search_space) {
            int difference = 0;
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < patchsize[0]; l++) {
                    for (int m = 0; m < patchsize[1]; m++) {
                        difference += abs(grid[v][j][k][pos + l][prev_position[1] + m] - grid[i][j][k][start_row + l][start_col + m]);
                    }
                }
            }
            if (difference < min_difference) {
                min_difference = difference;
                prev_position = {pos, prev_position[1]};
            }
        }
        vector<vector<vector<uint8_t>>> best_patch = vector<vector<vector<uint8_t>>>(3, vector<vector<uint8_t>>(
                patchsize[0], vector<uint8_t>(patchsize[1])));
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < patchsize[0]; l++) {
                for (int m = 0; m < patchsize[1]; m++) {
                    best_patch[k][l][m] = grid[v][j][k][prev_position[0] + l][prev_position[1] + m];
                }
            }
        }
        matching_patches = limited_insert(matching_patches, differences, best_patch, min_difference, num_similar);
    }
    return matching_patches;
}

vector<vector<vector<uint8_t>>> get_frankenpatches(vector<vector<vector<vector<vector<uint8_t>>>>> grid,
                                                   int i,
                                                   int j,
                                                   int patch_size,
                                                   int num_similar,
                                                   int search_stride,
                                                   int roi) {
    vector<vector<vector<uint8_t>>> output = vector<vector<vector<uint8_t>>>(3 * num_similar + 3,
                                                                             vector<vector<uint8_t>>(
                                                                                     grid[i][j][0].size(),
                                                                                     vector<uint8_t>(
                                                                                             grid[i][j][0][0].size())));
    vector<int> h_pos, w_pos;
    for (int h = 0; h < grid[i][j][0].size(); h += patch_size) {
        h_pos.emplace_back(h);
    }
    for (int w = 0; w < grid[i][j][0][0].size(); w += patch_size) {
        w_pos.emplace_back(w);
    }

    for (auto h: h_pos) {
        for (auto w: w_pos) {
            vector<vector<vector<uint8_t>>> matching_patches = get_matching_patches(grid,
                                                                                    i,
                                                                                    j,
                                                                                    h,
                                                                                    w,
                                                                                    patch_size,
                                                                                    num_similar,
                                                                                    search_stride,
                                                                                    roi);
            vector<int> patchsize = vector<int>(2, 0);
            patchsize[0] = min((int) grid[i][j][0].size() - h, patch_size);
            patchsize[1] = min((int) grid[i][j][0][0].size() - w, patch_size);
            vector<vector<vector<uint8_t>>> original_patch = vector<vector<vector<uint8_t>>>(3,
                                                                                             vector<vector<uint8_t>>(
                                                                                                     patchsize[0],
                                                                                                     vector<uint8_t>(
                                                                                                             patchsize[1])));
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < patchsize[0]; l++) {
                    for (int m = 0; m < patchsize[1]; m++) {
                        original_patch[k][l][m] = grid[i][j][k][h + l][w + m];
                    }
                }
            }
            matching_patches.emplace_back(original_patch[0]);
            matching_patches.emplace_back(original_patch[1]);
            matching_patches.emplace_back(original_patch[2]);
            while (matching_patches.size() < num_similar * 3 + 3) {
                matching_patches.emplace_back(original_patch[0]);
                matching_patches.emplace_back(original_patch[1]);
                matching_patches.emplace_back(original_patch[2]);
            }

            // write the matching patches back into output
            for (int k = 0; k < matching_patches.size(); k++) {
                for (int l = 0; l < matching_patches[0].size(); l++) {
                    for (int m = 0; m < matching_patches[0][0].size(); m++) {
                        output[k][h + l][w + m] = matching_patches[k][l][m];
                    }
                }
            }
        }
    }
    return output;
}

void save_data(vector<vector<vector<uint8_t>>> data, const string &filename) {
    vector<int> flat_data = vector<int>(data.size() * data[0].size() * data[0][0].size());
    for (int i = 0; i < data[0].size(); i++) {
        for (int j = 0; j < data[0][0].size(); j++) {
            for (int k = 0; k < data.size(); k++) {
                flat_data[i * data[0][0].size() * data.size() + j * data.size() + k] = (int) data[k][i][j];
            }
        }
    }

    cnpy::npy_save(filename,
                   &flat_data[0],
                   {static_cast<unsigned long>(data[0].size()),
                    static_cast<unsigned long>(data[0][0].size()),
                    static_cast<unsigned long>(data.size())},
                   "w");
}
