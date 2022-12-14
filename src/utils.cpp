//
// Created by tsfeith on 13/12/22.
//

#include <filesystem>
#include <vector>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include "cnpy.h"

using namespace std;

vector<string> get_scene_names(const string& scene_dir) {
    vector<string> scene_names;
    for (const auto &entry : filesystem::directory_iterator(scene_dir)) {
        if (entry.path().extension() != ".png") {
            continue;
        }
        scene_names.push_back(entry.path().filename());
    }
    sort(scene_names.begin(), scene_names.end());
    return scene_names;
}

vector<vector<vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>>>> get_scene_grid(const string& directory_path) {
    vector<vector<vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>>>> scene_grid;
    vector<filesystem::directory_entry> entries;
    for (const auto &entry : filesystem::directory_iterator(directory_path)) {
        // check if extension is ".png"
        if (entry.path().extension() != ".png") {
            continue;
        }
        entries.push_back(entry);
    }
    // sort entries by filename
    sort(entries.begin(), entries.end(), [](const filesystem::directory_entry& a, const filesystem::directory_entry& b) {
        return a.path().filename() < b.path().filename();
    });
    for (const auto &entry : entries) {
        // images have format "SCENE_00_00.png" ... "SCENE_XX_YY.png"
        // they should be stored in a vector with shape (XX, YY, 3, H, W)
        // where H and W are the height and width of the image
        // and 3 is the number of channels (RGB)
        string filename = entry.path().filename().string();
        int row = stoi(filename.substr(filename.length() - 9, 2));
        if (row == scene_grid.size()) {
            scene_grid.emplace_back();
        }
        // read the image using opencv
        cv::Mat image = cv::imread(entry.path().string());

        vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> data = vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>>(3, Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(image.rows, image.cols));
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                data[0](i, j) = image.at<cv::Vec3b>(i, j)[2];
                data[1](i, j) = image.at<cv::Vec3b>(i, j)[1];
                data[2](i, j) = image.at<cv::Vec3b>(i, j)[0];
            }
        }

        scene_grid[row].push_back(data);
    }
    return scene_grid;
}

long long int absoluteDifference(vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> img1, vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> img2) {
    long long int sum = 0;
    for (int i = 0; i < img1.size(); i++) {
        for (int j = 0; j < img1[i].rows(); j++) {
            for (int k = 0; k < img1[i].cols(); k++) {
                sum += abs((long long int)(img1[i](j, k)) - (long long int)(img2[i](j, k)));
            }
        }
    }
    return sum;
}

vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> limited_insert(vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> matching_patches,
                                       vector<long long int>& differences,
                                       vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> best_patch,
                                       long long int min_difference,
                                       int num_similar) {
    if (differences.size() < num_similar) {
        matching_patches.emplace_back(best_patch[0]);
        matching_patches.emplace_back(best_patch[1]);
        matching_patches.emplace_back(best_patch[2]);
        differences.emplace_back(min_difference);
        if (differences.size() == num_similar) {
            // sort differences and matching_patches jointly
            // in ascending order of differences
            for (int i = 0; i < differences.size(); i++) {
                for (int j = i + 1; j < differences.size(); j++) {
                    if (differences[i] > differences[j]) {
                        long long int temp = differences[i];
                        differences[i] = differences[j];
                        differences[j] = temp;
                        Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic> temp_patch1 = matching_patches[3*i];
                        Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic> temp_patch2 = matching_patches[3*i+1];
                        Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic> temp_patch3 = matching_patches[3*i+2];
                        matching_patches[3*i] = matching_patches[3*j];
                        matching_patches[3*i+1] = matching_patches[3*j+1];
                        matching_patches[3*i+2] = matching_patches[3*j+2];
                        matching_patches[3*j] = temp_patch1;
                        matching_patches[3*j+1] = temp_patch2;
                        matching_patches[3*j+2] = temp_patch3;
                    }
                }
            }
        }
    }
    else {
        // if the new difference is bigger than the largest difference, just return the old matching_patches
        if (min_difference >= differences[num_similar - 1]) {
            return matching_patches;
        }
        // otherwise, insert it and the best match via binary search
        int left = 0;
        int right = num_similar - 1;
        while (left < right) {
            int mid = (left + right) / 2;
            if (differences[mid] < min_difference) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        // insert the new difference and patch at the left index
        differences.insert(differences.begin() + left, min_difference);
        matching_patches.insert(matching_patches.begin() + 3 * left, best_patch[0]);
        matching_patches.insert(matching_patches.begin() + 3 * left + 1, best_patch[1]);
        matching_patches.insert(matching_patches.begin() + 3 * left + 2, best_patch[2]);
        // remove the last element
        differences.pop_back();
        matching_patches.pop_back();
        matching_patches.pop_back();
        matching_patches.pop_back();
    }
    return matching_patches;
}

vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> get_matching_patches(vector<vector<vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>>>> grid,
                                             int i,
                                             int j,
                                             int start_row,
                                             int start_col,
                                             int patch_size,
                                             int num_similar,
                                             int search_stride,
                                             int roi) {

    vector<int> patchsize = vector<int>(2, 0);
    patchsize[0] = min((int)grid[i][j][0].rows() - start_row, patch_size);
    patchsize[1] = min((int)grid[i][j][0].cols() - start_col, patch_size);
    vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> target_patch = vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>>(3,
                                                                   Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(patchsize[0], patchsize[1]));
    for (int k = 0; k < 3; k++) {
        target_patch[k] = grid[i][j][k].block(start_row, start_col, patchsize[0], patchsize[1]);
    }

    vector<int> grid_h, grid_v;
    for (int k = 0; k < grid.size(); k++) {
        if (k == i) {
            continue;
        }
        grid_v.push_back(k);
    }
    for (int k = 0; k < grid[i].size(); k++) {
        if (k == j) {
            continue;
        }
        grid_h.push_back(k);
    }

    vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> matching_patches;
    vector<long long int> differences;
    vector<int> prev_position = {start_row, start_col};
    for (auto h: grid_h) {
        long long int min_difference = 1e16;
        vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> best_patch = vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>>(3, Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(patchsize[0], patchsize[1]));
        vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> new_image = grid[i][h];
        vector<vector<int>> search_space;
        for (int k = -roi; k <=roi; k++) {
            vector<int> range_h = {prev_position[1] + k * search_stride,
                                   prev_position[1] + k * search_stride + patchsize[1]};
            if (range_h[0] >= 0 and range_h[1] <= new_image[0].cols()) {
                search_space.emplace_back(range_h);
            }
        }

        for (auto pos : search_space) {
            // create candidate patch using block from new image
            vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> candidate_patch = vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>>(3,
                                                                              Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(patchsize[0], patchsize[1]));
            for (int k = 0; k < 3; k++) {
                candidate_patch[k] = new_image[k].block(prev_position[0], pos[0], patchsize[0], patchsize[1]);
            }
            long long int difference = absoluteDifference(target_patch, candidate_patch);
            if (difference < min_difference) {
                min_difference = difference;
                best_patch = candidate_patch;
                prev_position = {prev_position[0], pos[0]};
            }
        }
        matching_patches = limited_insert(matching_patches, differences, best_patch, min_difference, num_similar);
    }

    for (auto v : grid_v) {
        long long int min_difference = 1e16;
        vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> best_patch = vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>>(3, Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(patchsize[0], patchsize[1]));
        vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> new_image = grid[v][j];
        vector<vector<int>> search_space;
        for (int k = -roi; k <=roi; k++) {
            vector<int> range_v = {prev_position[0] + k * search_stride,
                                   prev_position[0] + k * search_stride + patchsize[0]};
            if (range_v[0] >= 0 and range_v[1] <= new_image[0].rows()) {
                search_space.emplace_back(range_v);
            }
        }

        for (auto pos : search_space) {
            // create candidate patch using block from new image
            vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> candidate_patch = vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>>(3,
                                                                              Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(patchsize[0], patchsize[1]));
            for (int k = 0; k < 3; k++) {
                candidate_patch[k] = new_image[k].block(pos[0], prev_position[1], patchsize[0], patchsize[1]);
            }
            long long int difference = absoluteDifference(target_patch, candidate_patch);
            if (difference < min_difference) {
                min_difference = difference;
                best_patch = candidate_patch;
                prev_position = {pos[0], prev_position[1]};
            }
        }
        matching_patches = limited_insert(matching_patches, differences, best_patch, min_difference, num_similar);
    }
    return matching_patches;
}

vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> get_frankenpatches(vector<vector<vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>>>> grid,
                                           int i,
                                           int j,
                                           int patch_size,
                                           int num_similar,
                                           int search_stride,
                                           int roi) {
    vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> output = vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>>(3 * num_similar,
                                                            Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(grid[i][j][0].rows(), grid[i][j][0].cols()));

    vector<int> h_pos, w_pos;
    for (int h = 0; h < grid[i][j][0].rows(); h += patch_size) {
        h_pos.push_back(h);
    }
    for (int w = 0; w < grid[i][j][0].cols(); w += patch_size) {
        w_pos.push_back(w);
    }

    for (auto h : h_pos) {
        for (auto w : w_pos) {
            vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> matching_patches = get_matching_patches(grid, i, j, h, w, patch_size, num_similar-1, search_stride, roi);
            // insert the original patch into the matching patches
            vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> original_patch = vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>>(3,
                                                                            Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(patch_size, patch_size));
            for (int k = 0; k < 3; k++) {
                original_patch[k] = grid[i][j][k].block(h, w, matching_patches[0].rows(), matching_patches[0].cols());
            }
            // add the original patch to the matching patches, at the front
            matching_patches.insert(matching_patches.begin(), original_patch[2]);
            matching_patches.insert(matching_patches.begin(), original_patch[1]);
            matching_patches.insert(matching_patches.begin(), original_patch[0]);

            // while the size of the matching patches is less than num_similar, add the original patch
            while (matching_patches.size() < num_similar * 3) {
                matching_patches.insert(matching_patches.begin(), original_patch[2]);
                matching_patches.insert(matching_patches.begin(), original_patch[1]);
                matching_patches.insert(matching_patches.begin(), original_patch[0]);
            }

            // write the matching patches back into output
            for (int k = 0; k < num_similar; k++) {
                output[3 * k].block(h, w, matching_patches[3 * k].rows(), matching_patches[3 * k].cols()) = matching_patches[3 * k];
                output[3 * k + 1].block(h, w, matching_patches[3 * k + 1].rows(), matching_patches[3 * k + 1].cols()) = matching_patches[3 * k + 1];
                output[3 * k + 2].block(h, w, matching_patches[3 * k + 2].rows(), matching_patches[3 * k + 2].cols()) = matching_patches[3 * k + 2];
            }
        }
    }
    return output;
}

void save_data(vector<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> data, const string& filename) {
    vector<uint8_t> flat_data = vector<uint8_t>(data[0].rows() * data[0].cols() * data.size());
    for (int i = 0; i < data[0].rows(); i++) {
        for (int j = 0; j < data[0].cols(); j++) {
            for (int k = 0; k < data.size(); k++) {
                flat_data[i * data[0].cols() * data.size() + j * data.size() + k] = data[k](i, j);
            }
        }
    }

    cnpy::npy_save(filename,
                   &flat_data[0],
                   {static_cast<unsigned long>(data[0].rows()),static_cast<unsigned long>(data[0].cols()),data.size()},
                   "w");
}