//
// Created by tsfeith on 13/12/22.
//

#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>
#include "cnpy.h"

using namespace std;

vector<string> get_scene_names(const string &scene_dir, int grid_size_0, int grid_size_1) {
    vector<string> scene_names;
    for (const auto &entry: filesystem::directory_iterator(scene_dir)) {
        // only consider the "png" files
        if (entry.path().extension() != ".png") {
            continue;
        }
        string filename = entry.path().filename();
        // ignore the view outside of the subgrid selected
        if (stoi(filename.substr(filename.size() - 6, 2)) >= grid_size_1) {
            continue;
        }
        if (stoi(filename.substr(filename.size() - 9, 2)) >= grid_size_0) {
            continue;
        }
        scene_names.push_back(entry.path().filename());
    }
    sort(scene_names.begin(), scene_names.end());
    return scene_names;
}

vector<vector<vector<vector<vector<uint8_t>>>>> get_scene_grid(const string &directory_path,
                                                               int grid_size_0,
                                                               int grid_size_1) {
    vector<vector<vector<vector<vector<uint8_t>>>>> scene_grid;
    vector<filesystem::directory_entry> entries;
    // start by getting all the png files in the directory
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

    // take the sorted filenames, and get them into a grid of size grid_size_0 x grid_size_1
    for (const auto &entry: entries) {
        string filename = entry.path().filename().string();
        int row = stoi(filename.substr(filename.length() - 9, 2));
        if (row == scene_grid.size()) {
            if (scene_grid.size() == grid_size_0) {
                break;
            }
            scene_grid.emplace_back();
        }
        if (scene_grid[row].size() == grid_size_1) {
            continue;
        }
        // use openCV to read the image
        cv::Mat image = cv::imread(entry.path().string());

        // openCV uses the colour space BGR, so we need to convert it to RGB
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

void limited_insert(vector<vector<int>> &matching_patches,
                    vector<uint8_t> &differences,
                    const vector<int> &best_patch,
                    int min_diff,
                    int max_size) {
    // if the differences vector is not full yet, just add the new patch
    if (differences.size() < max_size) {
        matching_patches.emplace_back(best_patch);
        differences.push_back(min_diff);
        // check to see if it is now full
        // if it is, sort the differences vector and the matching patches vector jointly
        // we're using a bubble sort here, but it should be fine since the vector is small (~<10)
        if (differences.size() == max_size) {
            for (int i = 0; i < differences.size(); i++) {
                for (int j = i + 1; j < differences.size(); j++) {
                    if (differences[i] > differences[j]) {
                        swap(differences[i], differences[j]);
                        swap(matching_patches[i], matching_patches[j]);
                    }
                }
            }
        }
    } else {
        // if the differences vector is full, check to see if the new patch is better than the worst one
        if (min_diff >= differences[max_size - 1]) {
            return;
        }
        // if it is, put it in the right place in the vector using binary search (since the vector is sorted)
        differences.pop_back();
        matching_patches.pop_back();
        int left = 0;
        int right = max_size - 2;
        while (left < right) {
            int mid = (left + right) / 2;
            if (differences[mid] < min_diff) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        differences.insert(differences.begin() + left, min_diff);
        matching_patches.insert(matching_patches.begin() + left, best_patch);
    }
}

vector<vector<int>> get_matching_patches(vector<vector<vector<vector<vector<uint8_t>>>>> grid,
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

    vector<int> imgs_left, imgs_right, imgs_up, imgs_down;
    for (int k = i + 1; k < grid.size(); k++) {
        imgs_down.push_back(k);
    }
    for (int k = i - 1; k >= 0; k--) {
        imgs_up.push_back(k);
    }
    for (int k = j + 1; k < grid[i].size(); k++) {
        imgs_right.push_back(k);
    }
    for (int k = j - 1; k >= 0; k--) {
        imgs_left.push_back(k);
    }

    vector<vector<int>> matching_patches;
    vector<uint8_t> differences;
    vector<int> prev_position = {start_row, start_col};

    for (auto h: imgs_right) {
        uint8_t min_difference = 255;
        for (int a = -roi; a <= roi; a++) {
            if (prev_position[1] + a * search_stride < 0 or
                prev_position[1] + a * search_stride + patchsize[1] > grid[i][h][0][0].size()) {
                continue;
            }
            int pos = prev_position[1] + a * search_stride;
            int difference = 0;
            // get the L1 difference between the reference patch and the current patch
            // using new variables would make it easier to read, but it's faster to just access the grid
            for (int l = 0; l < patchsize[0]; l++) {
                for (int m = 0; m < patchsize[1]; m++) {
                    difference += abs(grid[i][h][0][prev_position[0] + l][pos + m] -
                                      grid[i][j][0][start_row + l][start_col + m]);
                    difference += abs(grid[i][h][1][prev_position[0] + l][pos + m] -
                                      grid[i][j][1][start_row + l][start_col + m]);
                    difference += abs(grid[i][h][2][prev_position[0] + l][pos + m] -
                                      grid[i][j][2][start_row + l][start_col + m]);
                }
            }
            difference /= 3 * patchsize[0] * patchsize[1];

            if (difference < min_difference) {
                min_difference = (uint8_t) difference;
                prev_position = {prev_position[0], pos};
            }
        }
        limited_insert(matching_patches, differences, {i, h, prev_position[0], prev_position[1]}, min_difference,
                       num_similar);
    }

    prev_position = {start_row, start_col};
    for (auto h: imgs_left) {
        int min_difference = 255;
        for (int a = -roi; a <= roi; a++) {
            if (prev_position[1] + a * search_stride < 0 or
                prev_position[1] + a * search_stride + patchsize[1] > grid[i][h][0][0].size()) {
                continue;
            }
            int pos = prev_position[1] + a * search_stride;
            int difference = 0;
            for (int l = 0; l < patchsize[0]; l++) {
                for (int m = 0; m < patchsize[1]; m++) {
                    difference += abs(
                            grid[i][h][0][prev_position[0] + l][pos + m] - grid[i][j][0][start_row + l][start_col + m]);
                    difference += abs(
                            grid[i][h][1][prev_position[0] + l][pos + m] - grid[i][j][1][start_row + l][start_col + m]);
                    difference += abs(
                            grid[i][h][2][prev_position[0] + l][pos + m] - grid[i][j][2][start_row + l][start_col + m]);
                }
            }
            difference /= 3 * patchsize[0] * patchsize[1];
            if (difference < min_difference) {
                min_difference = (uint8_t) difference;
                prev_position = {prev_position[0], pos};
            }
        }
        limited_insert(matching_patches, differences, {i, h, prev_position[0], prev_position[1]}, min_difference,
                       num_similar);
    }

    prev_position = {start_row, start_col};
    for (auto h: imgs_down) {
        int min_difference = 255;
        for (int a = -roi; a <= roi; a++) {
            if (prev_position[0] + a * search_stride < 0 or
                prev_position[0] + a * search_stride + patchsize[0] > grid[h][j][0].size()) {
                continue;
            }
            int pos = prev_position[0] + a * search_stride;
            int difference = 0;
            for (int l = 0; l < patchsize[0]; l++) {
                for (int m = 0; m < patchsize[1]; m++) {
                    difference += abs(
                            grid[i][h][0][prev_position[0] + l][pos + m] - grid[i][j][0][start_row + l][start_col + m]);
                    difference += abs(
                            grid[i][h][1][prev_position[0] + l][pos + m] - grid[i][j][1][start_row + l][start_col + m]);
                    difference += abs(
                            grid[i][h][2][prev_position[0] + l][pos + m] - grid[i][j][2][start_row + l][start_col + m]);
                }
            }
            difference /= 3 * patchsize[0] * patchsize[1];
            if (difference < min_difference) {
                min_difference = (uint8_t) difference;
                prev_position = {pos, prev_position[1]};
            }
        }
        limited_insert(matching_patches, differences, {h, j, prev_position[0], prev_position[1]}, min_difference,
                       num_similar);
    }

    prev_position = {start_row, start_col};
    for (auto h: imgs_up) {
        int min_difference = 255;
        for (int a = -roi; a <= roi; a++) {
            if (prev_position[0] + a * search_stride < 0 or
                prev_position[0] + a * search_stride + patchsize[0] > grid[h][j][0].size()) {
                continue;
            }
            int pos = prev_position[0] + a * search_stride;
            int difference = 0;
            for (int l = 0; l < patchsize[0]; l++) {
                for (int m = 0; m < patchsize[1]; m++) {
                    difference += abs(
                            grid[i][h][0][prev_position[0] + l][pos + m] - grid[i][j][0][start_row + l][start_col + m]);
                    difference += abs(
                            grid[i][h][1][prev_position[0] + l][pos + m] - grid[i][j][1][start_row + l][start_col + m]);
                    difference += abs(
                            grid[i][h][2][prev_position[0] + l][pos + m] - grid[i][j][2][start_row + l][start_col + m]);
                }
            }
            if (difference < min_difference) {
                min_difference = (uint8_t) difference;
                prev_position = {pos, prev_position[1]};
            }
        }
        limited_insert(matching_patches, differences, {h, j, prev_position[0], prev_position[1]}, min_difference,
                       num_similar);
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
    for (int h = 0; h < grid[i][j][0].size(); h += patch_size) {
        for (int w = 0; w < grid[i][j][0][0].size(); w += patch_size) {
            vector<vector<int>> matching_patches = get_matching_patches(grid,
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

            matching_patches.emplace_back(vector<int>{i, j, h, w});
            while (matching_patches.size() < num_similar + 1) {
                matching_patches.emplace_back(vector<int>{i, j, h, w});
            }

            // write the matching patches into output
            for (int k = 0; k < matching_patches.size(); k++) {
                for (int l = 0; l < patchsize[0]; l++) {
                    for (int m = 0; m < patchsize[1]; m++) {
                        for (int n = 0; n < 3; n++) {
                            output[k * 3 + n][h + l][w + m] = grid[matching_patches[k][0]][matching_patches[k][1]][n][
                                    matching_patches[k][2] + l][matching_patches[k][3] + m];
                        }
                    }
                }
            }
        }
    }
    return output;
}

void save_data(vector<vector<vector<uint8_t>>> data, const string &filename) {
    vector<uint8_t> flat_data = vector<uint8_t>(data.size() * data[0].size() * data[0][0].size());
    for (int i = 0; i < data[0].size(); i++) {
        for (int j = 0; j < data[0][0].size(); j++) {
            for (int k = 0; k < data.size(); k++) {
                flat_data[i * data[0][0].size() * data.size() + j * data.size() + k] = data[k][i][j];
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
